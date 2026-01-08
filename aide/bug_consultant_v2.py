"""
Bug Consultant v2: Memory Writer + Retrieval-Only Reader for Debugging History

Design goals (ICML submission aligned):
- In-context RL (no parameter updates): improve behavior via better context and action selection.
- World model writing: persist compact, policy-relevant "what works / what fails" rules (not raw logs).
- Separation of retrieval and execution: provide curated context to the code-generating actor.

This module is adapted from `references/aideml_vm/aide/bug_consultant_v2.py` with:
- stronger trial tracking (records failed trials instead of starting new bug records)
- deterministic world model writing (no extra LLM call required)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .backend import FunctionSpec, query

if TYPE_CHECKING:
    from .journal import Node, Journal

logger = logging.getLogger("aide")


# ═══════════════════════════════════════════════════════════
# Data Structures (Simple Storage)
# ═══════════════════════════════════════════════════════════


@dataclass
class DebugTrial:
    """Record of a single debug attempt on a parent bug."""

    attempt_num: int
    node_step: int
    debug_plan: str
    code: str
    outcome: str  # "success" | "failed"
    error_type: Optional[str] = None
    error_output: Optional[str] = None
    why_worked: Optional[str] = None
    why_failed: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class BugRecord:
    """Organized record of a bug and all attempts to fix it."""

    bug_id: str
    original_node_step: int
    error_type: str
    buggy_code: str
    buggy_output: str
    original_plan: str

    trials: list[DebugTrial] = field(default_factory=list)
    final_outcome: str = "in_progress"  # "success" | "abandoned" | "in_progress"

    # LLM-summarized fields at bug start (for RAG)
    error_signature: Optional[str] = None  # Specific error message (LLM-extracted)
    error_category: Optional[str] = None  # Category for grouping (API_MISUSE, etc.)
    initial_hypothesis: Optional[str] = None  # Preliminary root cause
    context_tags: list[str] = field(default_factory=list)  # Tags for semantic matching

    # Extracted / summarized fields (from trials and completion)
    root_cause: Optional[str] = None  # Final confirmed root cause
    successful_strategy: Optional[str] = None  # What worked
    failed_strategies: list[str] = field(default_factory=list)  # What didn't work (incremental)
    learned_constraints: list[str] = field(default_factory=list)  # Reusable rules from failures
    lesson: Optional[str] = None  # Actionable takeaway

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


# ═══════════════════════════════════════════════════════════
# Function Specs for LLM Calls
# ═══════════════════════════════════════════════════════════


retrieve_spec = FunctionSpec(
    name="retrieve_optimal_context",
    json_schema={
        "type": "object",
        "properties": {
            "selected_bug_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Bug IDs to retrieve - choose an optimal number (not fixed).",
            },
            "reasoning": {
                "type": "string",
                "description": "Why these bugs? Why this number?",
            },
            "key_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key patterns across selected bugs",
            },
        },
        "required": ["selected_bug_ids", "reasoning", "key_patterns"],
    },
    description="Select the most relevant historical bugs for the current issue and explain the reasoning and shared patterns.",
)

summarize_bug_record_spec = FunctionSpec(
    name="summarize_bug_record",
    json_schema={
        "type": "object",
        "properties": {
            "root_cause": {
                "type": "string",
                "description": "The ACTUAL root cause explanation - what technically caused this bug. Be specific and concise (1-2 sentences). Example: 'LGBMRegressor.fit() doesn't accept early_stopping_rounds parameter in sklearn API v3.0+, requires callbacks parameter instead'. NOT just 'TypeError' or repeating the error message.",
            },
            "successful_strategy": {
                "type": "string",
                "description": "Specific strategy that worked, in one concise sentence. Example: 'Use callbacks=[lgb.early_stopping(50)] instead of early_stopping_rounds parameter'. Focus on the actionable fix, not verbose explanation.",
            },
            "failed_strategies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific strategies that failed with WHY each failed. One sentence per strategy. Example: 'Tried early_stopping_rounds parameter → incompatible with sklearn API'. Keep each entry concise but informative.",
            },
            "lesson": {
                "type": "string",
                "description": "Actionable, reusable pattern that can prevent this exact mistake in future. One concise sentence. Example: 'For LightGBM sklearn API TypeError with early_stopping_rounds: use callbacks=[lgb.early_stopping(rounds)] instead'. Must be specific enough that seeing this lesson prevents repeating the mistake.",
            },
        },
        "required": ["root_cause", "lesson"],
    },
    description="Extract root cause, strategies, and reusable lessons from a completed bug record. Be SPECIFIC and CONCISE - the goal is to create actionable knowledge that prevents repeating the same mistakes.",
)

summarize_bug_start_spec = FunctionSpec(
    name="summarize_bug_start",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "Compact, searchable error signature capturing the SPECIFIC error message for semantic matching. Example: 'TypeError: LGBMRegressor.fit() got unexpected keyword argument early_stopping_rounds'. Be precise and concise - include the error type and the key message that differentiates this from other similar errors.",
            },
            "error_category": {
                "type": "string",
                "description": "High-level category for grouping: API_MISUSE (wrong API usage), MISSING_DATA (NaN/missing values), TYPE_ERROR (type mismatch), IMPORT_ERROR (module not found), LOGIC_ERROR (wrong logic), VALUE_ERROR (invalid value), ATTRIBUTE_ERROR (missing attribute), KEY_ERROR (missing key), INDEX_ERROR (out of bounds), or OTHER",
            },
            "initial_hypothesis": {
                "type": "string",
                "description": "Preliminary root cause hypothesis - what likely caused this error based on the signature and output. Be specific about the technical reason. Example: 'LightGBM sklearn API changed - early_stopping_rounds parameter was removed in newer versions, now requires callbacks=[lgb.early_stopping()] instead'.",
            },
            "context_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-7 relevant context tags for RAG semantic matching. Include: library names, API names, error-specific keywords, domain terms. Example: ['lightgbm', 'sklearn_api', 'early_stopping', 'parameter_deprecation']. Use lowercase, underscores for multi-word tags.",
            },
        },
        "required": ["error_signature", "error_category", "initial_hypothesis", "context_tags"],
    },
    description="Summarize a new bug at start for RAG retrieval. Keep outputs concise but informative - error_signature should be one line, hypothesis should be 1-2 sentences.",
)

summarize_trial_failure_spec = FunctionSpec(
    name="summarize_trial_failure",
    json_schema={
        "type": "object",
        "properties": {
            "why_failed": {
                "type": "string",
                "description": "Specific reason WHY this debug approach failed - focus on the ROOT CAUSE of failure, not just repeating the error type. Be concise (1-2 sentences). Example: 'Model still expects early_stopping_rounds despite using callbacks - incorrect callback API usage, needs eval_set parameter too'.",
            },
            "failed_strategy_summary": {
                "type": "string",
                "description": "One-line summary of the failed strategy for RL learning. Format: '<what was tried> → Failed: <why>'. Example: 'Use callbacks without eval_set → Failed: callbacks require validation data'. Keep concise.",
            },
            "learned_constraint": {
                "type": "string",
                "description": "A reusable constraint/rule learned from this failure - one concise sentence that prevents repeating this mistake. Example: 'LightGBM callbacks require both callbacks=[] AND eval_set=[(X_val, y_val)] parameters together'.",
            },
        },
        "required": ["why_failed", "failed_strategy_summary", "learned_constraint"],
    },
    description="Summarize a failed debug trial for RL. Extract specific failure reason, strategy summary, and learned constraint to prevent repeating this mistake. Be concise but informative.",
)

summarize_trial_success_spec = FunctionSpec(
    name="summarize_trial_success",
    json_schema={
        "type": "object",
        "properties": {
            "why_worked": {
                "type": "string",
                "description": "Specific reason WHY this approach succeeded. Be concise (1-2 sentences). Example: 'Using callbacks=[lgb.early_stopping(50)] with eval_set parameter correctly implements early stopping in LightGBM sklearn API'.",
            },
            "successful_strategy_summary": {
                "type": "string",
                "description": "One-line summary of successful strategy - what fixed the bug. Example: 'Replace early_stopping_rounds with callbacks=[lgb.early_stopping(rounds)] + eval_set'. Keep concise and actionable.",
            },
            "key_insight": {
                "type": "string",
                "description": "Key insight that made this work - the core understanding. One concise sentence. Example: 'LightGBM sklearn API requires callback-based early stopping with explicit validation set'.",
            },
        },
        "required": ["why_worked", "successful_strategy_summary", "key_insight"],
    },
    description="Summarize a successful debug trial for RL. Extract why it worked, strategy summary, and key insight. Be concise but complete.",
)

extract_error_signature_spec = FunctionSpec(
    name="extract_error_signature",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "Compact, specific error signature. Example: 'TypeError: fit() got an unexpected keyword argument early_stopping_rounds'. One line, be precise.",
            },
            "error_category": {
                "type": "string",
                "description": "High-level category: API_MISUSE, MISSING_DATA, TYPE_ERROR, IMPORT_ERROR, LOGIC_ERROR, VALUE_ERROR, ATTRIBUTE_ERROR, KEY_ERROR, INDEX_ERROR, or OTHER",
            },
        },
        "required": ["error_signature", "error_category"],
    },
    description="Extract a meaningful error signature and category from execution output for bug tracking.",
)

format_context_spec = FunctionSpec(
    name="format_debug_context",
    json_schema={
        "type": "object",
        "properties": {
            "formatted_context": {
                "type": "string",
                "description": (
                    "Organized context in clean markdown format with sections: "
                    "Overview, What Worked, What Failed, Key Patterns, Lessons."
                ),
            },
            "summary": {
                "type": "string",
                "description": "One paragraph summary of key insights.",
            },
        },
        "required": ["formatted_context", "summary"],
    },
    description="Organize selected historical bugs into a clear markdown context for the actor.",
)


# ═══════════════════════════════════════════════════════════
# Bug Consultant Class
# ═══════════════════════════════════════════════════════════


class BugConsultant:
    """
    Memory writer + retrieval-only reader for debugging history.

    Responsibilities:
    - Write: store bug records, trials, and a compact world model (markdown).
    - Read: retrieve and format relevant historical bugs as curated context.

    Not responsible for:
    - Writing code (that's the actor's job).
    - Deciding what to do (search policy remains in the agent).
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.3,
        save_dir: Optional[Path] = None,
        max_bug_records: int = 50,
        advice_budget_chars: int = 200000,
        max_active_bugs: int = 200,
        max_trials_per_bug: int = 20,
        delete_pruned_bug_files: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_bug_records = max_bug_records
        self.advice_budget_chars = int(advice_budget_chars)
        self.max_active_bugs = max_active_bugs
        self.max_trials_per_bug = max_trials_per_bug
        self.delete_pruned_bug_files = delete_pruned_bug_files

        self.save_dir = save_dir
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.bug_records: dict[str, BugRecord] = {}
        self.active_bugs: dict[str, BugRecord] = {}

        # World model version counter (increments on write)
        self.world_model_version: int = 0

        logger.info("🗂️  BugConsultant v2 initialized (model: %s)", model)

    # ═══════════════════════════════════════════════════════════
    # Writer: Bug record lifecycle
    # ═══════════════════════════════════════════════════════════

    def start_bug_record(self, node: "Node") -> str:
        if node.step is None:
            raise ValueError("Cannot start bug record: node.step is None")

        self._enforce_active_bug_limit()

        bug_id = f"bug_{node.step}"
        # Use full output (not truncated) for bug analysis
        full_output = "".join(node._term_out) if hasattr(node, "_term_out") else node.term_out
        record = BugRecord(
            bug_id=bug_id,
            original_node_step=node.step,
            error_type=node.exc_type or "Unknown",
            buggy_code=node.code,
            buggy_output=full_output,
            original_plan=node.plan or "",
        )

        # Stage 1: LLM summarization at bug start (for RAG)
        logger.debug("Summarizing bug start for %s", bug_id)
        summary = self._summarize_bug_start(record)
        record.error_signature = summary.get("error_signature")
        record.error_category = summary.get("error_category")
        record.initial_hypothesis = summary.get("initial_hypothesis")
        record.context_tags = summary.get("context_tags", [])

        self.active_bugs[bug_id] = record
        logger.info("📝 Started bug record: %s (category: %s)", bug_id, record.error_category)
        self._save_bug_record(record, active=True)
        return bug_id

    def record_trial(
        self,
        bug_id: str,
        node: "Node",
        outcome: str,
        why_worked: Optional[str] = None,
        why_failed: Optional[str] = None,
    ) -> None:
        if bug_id not in self.active_bugs:
            logger.warning("Bug %s not found in active bugs", bug_id)
            return
        if node.step is None:
            logger.warning("Skipping trial record: node.step is None")
            return

        record = self.active_bugs[bug_id]
        # Use full output (not truncated) for trial error analysis
        full_output = None
        if outcome == "failed":
            full_output = "".join(node._term_out) if hasattr(node, "_term_out") else node.term_out
        trial = DebugTrial(
            attempt_num=len(record.trials) + 1,
            node_step=node.step,
            debug_plan=node.plan or "",
            code=node.code,
            outcome=outcome,
            error_type=node.exc_type if outcome == "failed" else None,
            error_output=full_output,
            why_worked=why_worked,
            why_failed=why_failed,
        )
        record.trials.append(trial)

        # Stage 2: LLM summarization for incremental RL learning
        # This is THE KEY to preventing retrying failed strategies and reusing successful ones
        if outcome == "failed":
            logger.debug("Summarizing failed trial #%s for %s", trial.attempt_num, bug_id)
            summary = self._summarize_trial_failure(trial, record)

            # Store LLM-extracted insights in the trial
            trial.why_failed = summary.get("why_failed", trial.why_failed)

            # Add to failed strategies (for RL: prevent retrying)
            failed_strategy = summary.get("failed_strategy_summary", "")
            if failed_strategy and failed_strategy not in record.failed_strategies:
                record.failed_strategies.append(failed_strategy)

            # Add to learned constraints (reusable rules)
            constraint = summary.get("learned_constraint", "")
            if constraint and constraint not in record.learned_constraints:
                record.learned_constraints.append(constraint)

            logger.info("📊 Recorded FAILED trial #%s for %s: %s", trial.attempt_num, bug_id, failed_strategy)

        elif outcome == "success":
            logger.debug("Summarizing successful trial #%s for %s", trial.attempt_num, bug_id)
            summary = self._summarize_trial_success(trial, record)

            # Store LLM-extracted insights in the trial
            trial.why_worked = summary.get("why_worked", trial.why_worked)

            # Store successful strategy (for RL: reuse this approach)
            successful_strategy = summary.get("successful_strategy_summary", "")
            if successful_strategy:
                record.successful_strategy = successful_strategy

            logger.info("📊 Recorded SUCCESS trial #%s for %s: %s", trial.attempt_num, bug_id, successful_strategy)

        self._save_bug_record(record, active=True)

        # Safety valve: cap attempts per bug (rare; defaults are intentionally high).
        if self.max_trials_per_bug > 0 and len(record.trials) >= self.max_trials_per_bug:
            self._auto_abandon_bug_record(
                bug_id,
                reason=f"Exceeded max_trials_per_bug={self.max_trials_per_bug}",
            )

    def complete_bug_record(
        self, bug_id: str, outcome: str, *, journal: Optional["Journal"] = None
    ) -> None:
        if bug_id not in self.active_bugs:
            return

        record = self.active_bugs[bug_id]
        record.final_outcome = outcome

        # Extract lessons with one LLM call (fallback to minimal content).
        summary = self._summarize_bug_record(record)
        record.root_cause = summary.get("root_cause") or record.root_cause
        record.successful_strategy = summary.get("successful_strategy") or record.successful_strategy
        record.failed_strategies = summary.get("failed_strategies") or record.failed_strategies
        record.lesson = summary.get("lesson") or record.lesson

        self.bug_records[bug_id] = record
        self.active_bugs.pop(bug_id, None)
        self._prune_bug_records()

        logger.info("✅ Completed bug record %s", bug_id)
        self._save_bug_record(record, active=False)
        self._save_world_model(journal=journal)

    def abandon_bug_record(self, bug_id: str, *, journal: Optional["Journal"] = None) -> None:
        self.complete_bug_record(bug_id, outcome="abandoned", journal=journal)

    # ═══════════════════════════════════════════════════════════
    # Writer: Backward-compatible ingestion API
    # ═══════════════════════════════════════════════════════════

    def learn_from_bug(self, node: "Node", journal: Optional["Journal"] = None) -> None:
        """
        Update bug memory from a node outcome.

        Key behavior (fixes a weakness in the reference implementation):
        - If node has a buggy parent, treat it as a TRIAL for the parent bug (success or failure).
        - Only start a new bug record when the node is buggy AND its parent is not buggy.
        """
        if node.step is None:
            logger.warning("Skipping BugConsultant update: node.step is None")
            return

        # Trial path: any child of a buggy parent is a debug attempt on that parent bug.
        if node.parent is not None and getattr(node.parent, "is_buggy", False):
            parent_step = getattr(node.parent, "step", None)
            if parent_step is None:
                return
            parent_bug_id = f"bug_{parent_step}"
            if parent_bug_id not in self.active_bugs:
                # Ensure parent bug is tracked.
                try:
                    self.start_bug_record(node.parent)
                except Exception:
                    return

            if node.is_buggy:
                self.record_trial(
                    parent_bug_id,
                    node,
                    outcome="failed",
                    why_failed=node.analysis or (f"Still failing: {node.exc_type}" if node.exc_type else "Still failing"),
                )
                self._save_world_model(journal=journal)
                return

            self.record_trial(
                parent_bug_id,
                node,
                outcome="success",
                why_worked=node.analysis or "Executed successfully",
            )
            self.complete_bug_record(parent_bug_id, outcome="success", journal=journal)
            return

        # New bug: buggy node without a buggy parent.
        if node.is_buggy:
            self.start_bug_record(node)
            self._save_world_model(journal=journal)
            return

        # Non-buggy node without buggy parent: nothing to learn.
        self._save_world_model(journal=journal)

    def ingest_journal(self, journal: "Journal") -> None:
        """Rebuild memory state from an existing journal (resume mode)."""
        for node in journal.nodes:
            try:
                self.learn_from_bug(node, journal=journal)
            except Exception:
                continue

    # ═══════════════════════════════════════════════════════════
    # Reader: Retrieval and formatting (curated context only)
    # ═══════════════════════════════════════════════════════════

    def retrieve_relevant_context(
        self,
        current_error_type: str,
        current_error_msg: str,
        current_code: str,
        original_plan: str,
    ) -> dict:
        # Include BOTH completed AND active bugs for RAG
        # Active bugs show what was tried but didn't work yet - crucial for RL!
        if not self.bug_records and not self.active_bugs:
            return {"selected_bugs": [], "reasoning": "No historical bugs yet", "key_patterns": []}

        all_bugs = []

        # Add completed bugs
        for bug_id, record in self.bug_records.items():
            all_bugs.append(
                {
                    "bug_id": bug_id,
                    "status": "completed",  # Mark as completed
                    "error_type": record.error_type,
                    # RAG fields for semantic matching
                    "error_signature": record.error_signature,  # Specific error message
                    "error_category": record.error_category,  # API_MISUSE, TYPE_ERROR, etc.
                    "context_tags": record.context_tags,  # Semantic tags
                    "initial_hypothesis": record.initial_hypothesis,  # Preliminary root cause
                    # RL fields
                    "successful_strategy": record.successful_strategy,
                    "failed_strategies": record.failed_strategies,
                    "learned_constraints": record.learned_constraints,  # Reusable rules
                    # Completion fields
                    "root_cause": record.root_cause,
                    "lesson": record.lesson,
                    "outcome": record.final_outcome,
                    "trials_count": len(record.trials),
                }
            )

        # Add ACTIVE bugs (what's being tried right now but not working)
        for bug_id, record in self.active_bugs.items():
            all_bugs.append(
                {
                    "bug_id": bug_id,
                    "status": "active",  # Mark as active
                    "error_type": record.error_type,
                    # RAG fields for semantic matching
                    "error_signature": record.error_signature,
                    "error_category": record.error_category,
                    "context_tags": record.context_tags,
                    "initial_hypothesis": record.initial_hypothesis,
                    # RL fields - what didn't work so far
                    "failed_strategies": record.failed_strategies,  # Critical: what was tried and failed
                    "learned_constraints": record.learned_constraints,  # Rules learned from failures
                    "successful_strategy": None,  # Not solved yet
                    "root_cause": None,  # Not confirmed yet
                    "lesson": None,  # Not complete yet
                    "outcome": "in_progress",
                    "trials_count": len(record.trials),
                }
            )

        # Create a table of contents view for the LLM to scan
        bug_index_summary = []
        for bug in all_bugs:
            summary_line = (
                f"{bug['bug_id']} [{bug['status']}]: "
                f"{bug['error_category'] or 'Unknown'} - "
                f"{bug['error_signature'] or bug['error_type']} "
                f"(trials={bug['trials_count']})"
            )
            bug_index_summary.append(summary_line)

        prompt = {
            "Your Role": "RAG - Retrieve relevant bugs from historical memory",
            "Current Bug": {
                "Error Type": current_error_type,
                "Error Message": current_error_msg,
                "Code": current_code,
                "Plan Context": original_plan,
            },
            "Bug Index (Table of Contents)": bug_index_summary,
            "Detailed Bug Records": all_bugs,
            "How to Use This": (
                "1. SCAN the Bug Index to see all available bugs at a glance\n"
                "2. MATCH based on error signature, category, and tags\n"
                "3. SELECT the most relevant bugs\n\n"
                "Bug statuses:\n"
                "- COMPLETED: Has final solution and lessons\n"
                "- ACTIVE: Currently being debugged - shows what DIDN'T WORK (crucial for RL!)"
            ),
            "Task": (
                "Select an OPTIMAL set of relevant bug IDs based on semantic similarity.\n"
                "- Exact match: 1-2 bugs\n"
                "- Related patterns: 2-4 bugs\n"
                "- Unrelated: 0 bugs\n\n"
                "CRITICAL: Include ACTIVE bugs if similar - their failed_strategies show what to AVOID!"
            ),
        }

        try:
            result = query(
                system_message=prompt,
                user_message=None,
                func_spec=retrieve_spec,
                model=self.model,
                temperature=self.temperature,
            )
            bug_ids = result.get("selected_bug_ids", [])
            # Retrieve from BOTH completed AND active bugs
            selected = []
            for b in bug_ids:
                if b in self.bug_records:
                    selected.append(self.bug_records[b])
                elif b in self.active_bugs:
                    selected.append(self.active_bugs[b])
            return {
                "selected_bugs": selected,
                "reasoning": result.get("reasoning", ""),
                "key_patterns": result.get("key_patterns", []),
            }
        except Exception as e:
            logger.error("Bug retrieval failed: %s", e)
            # Fallback: match by error type from BOTH completed and active bugs
            matched = [
                r
                for r in list(self.bug_records.values()) + list(self.active_bugs.values())
                if (r.error_type or "").lower() == (current_error_type or "").lower()
            ]
            return {
                "selected_bugs": matched[:3],
                "reasoning": f"Fallback: matched by error type ({len(matched)} found)",
                "key_patterns": [],
            }

    def format_context_for_actor(self, retrieval_result: dict) -> str:
        if not retrieval_result.get("selected_bugs"):
            return "No relevant historical bugs found."

        bugs_data = []
        for record in retrieval_result["selected_bugs"]:
            bugs_data.append(
                {
                    "error_type": record.error_type,
                    "root_cause": record.root_cause,
                    "trials": [
                        {"attempt": t.attempt_num, "strategy": t.debug_plan, "outcome": t.outcome}
                        for t in record.trials
                    ],
                    "successful_strategy": record.successful_strategy,
                    "failed_strategies": record.failed_strategies,
                    "lesson": record.lesson,
                }
            )

        prompt = {
            "Your Role": "Format historical context for code generation",
            "Selected Bugs": bugs_data,
            "Key Patterns": retrieval_result.get("key_patterns", []),
            "Reasoning": retrieval_result.get("reasoning", ""),
            "Task": (
                "Format into clean, concise markdown.\n"
                "Sections:\n"
                "1) Overview\n"
                "2) What Worked\n"
                "3) What Failed\n"
                "4) Key Patterns\n"
                "5) Lessons\n"
                "Avoid verbosity; focus on actionable, reusable information."
            ),
        }

        try:
            result = query(
                system_message=prompt,
                user_message=None,
                func_spec=format_context_spec,
                model=self.model,
                temperature=0.3,
            )
            formatted = result.get("formatted_context", "") or ""
            # Safety valve: truncation is allowed ONLY for memory management.
            if self.advice_budget_chars > 0 and len(formatted) > self.advice_budget_chars:
                return formatted[: self.advice_budget_chars].rstrip()
            return formatted
        except Exception as e:
            logger.error("Context formatting failed: %s", e)
            lines = ["# Historical Context", ""]
            for record in retrieval_result["selected_bugs"]:
                lines.append(f"## {record.error_type}")
                if record.lesson:
                    lines.append(f"- Lesson: {record.lesson}")
                lines.append("")
            return "\n".join(lines).strip()

    def get_guidance(self, plan: str = "", current_node: Optional["Node"] = None) -> str:
        """
        Convenience wrapper (reference behavior): retrieve relevant bugs and format them.

        Note: this is still two LLM calls (retrieve + format), matching the reference implementation.
        """
        if not current_node:
            return ""
        if not self.bug_records:
            return "No historical bugs to learn from yet."

        try:
            # Use full output (not truncated) for better bug matching
            full_error_msg = "".join(current_node._term_out) if hasattr(current_node, "_term_out") else (current_node.term_out or "")
            retrieval = self.retrieve_relevant_context(
                current_error_type=current_node.exc_type or "Unknown",
                current_error_msg=full_error_msg,
                current_code=current_node.code,
                original_plan=plan,
            )
            if not retrieval.get("selected_bugs"):
                return "No relevant historical bugs found."
            return self.format_context_for_actor(retrieval)
        except Exception as e:
            logger.error("get_guidance failed: %s", e)
            return ""

    def get_prevention_guidance(self, mode: str = "executive") -> str:
        """
        Executive summary for prompts (reference-style): top recent lessons.

        This is a compact, prompt-safe view; detailed context should come from `get_guidance()`.
        """
        if not self.bug_records:
            return ""

        lessons = []
        # Most recent first
        for record in sorted(self.bug_records.values(), key=lambda r: r.timestamp, reverse=True):
            if record.lesson:
                lessons.append(f"- {record.error_type}: {record.lesson}")

        if not lessons:
            return ""

        header = f"📚 Historical Bug Lessons ({len(self.bug_records)} bugs tracked):"
        body = "\n".join(lessons[:5])  # Top 5 most recent
        out = f"{header}\n{body}".strip()
        # Safety valve: truncation is allowed ONLY for memory management.
        if mode == "executive" and self.advice_budget_chars > 0 and len(out) > self.advice_budget_chars:
            return out[: self.advice_budget_chars].rstrip()
        return out

    def get_statistics(self) -> dict:
        return {
            "total_bugs": len(self.bug_records),
            "active_bugs": len(self.active_bugs),
            "successful_fixes": sum(1 for r in self.bug_records.values() if r.final_outcome == "success"),
            "abandoned": sum(1 for r in self.bug_records.values() if r.final_outcome == "abandoned"),
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-based summarization (writer)
    # ═══════════════════════════════════════════════════════════

    def _summarize_bug_start(self, record: BugRecord) -> dict:
        """
        Stage 1: Summarize a new bug at start for RAG retrieval.
        Extracts error signature, category, hypothesis, and context tags.
        """
        prompt = {
            "Your Role": "Summarize a new bug for semantic retrieval",
            "Bug Context": {
                "Error Type": record.error_type,
                "Error Output": record.buggy_output,
                "Buggy Code": record.buggy_code,
                "Original Plan": record.original_plan,
            },
            "Task": (
                "Extract structured information for RAG matching:\n\n"
                "1. ERROR SIGNATURE: Specific error message (not generic 'Traceback...')\n"
                "2. ERROR CATEGORY: High-level category (API_MISUSE, TYPE_ERROR, etc.)\n"
                "3. INITIAL HYPOTHESIS: Preliminary root cause (what likely caused this)\n"
                "4. CONTEXT TAGS: 3-7 relevant tags for semantic matching\n\n"
                "This information helps match similar bugs later."
            ),
        }

        try:
            return query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_bug_start_spec,
                model=self.model,
                temperature=0.0,  # Deterministic for categorization
            )
        except Exception as e:
            logger.error("Failed to summarize bug start: %s", e)
            # Fallback to heuristic
            sig = self._error_signature(record.buggy_output)
            return {
                "error_signature": sig or f"{record.error_type}: (no signature extracted)",
                "error_category": "UNKNOWN",
                "initial_hypothesis": f"Bug occurred during execution with error type {record.error_type}",
                "context_tags": [record.error_type.lower().replace(" ", "_")] if record.error_type else [],
            }

    def _summarize_trial_failure(self, trial: DebugTrial, record: BugRecord) -> dict:
        """
        Stage 2a: Summarize a failed debug trial for RL.
        Extracts why it failed, strategy summary, and learned constraint.
        """
        prompt = {
            "Your Role": "Analyze why this debug attempt failed",
            "Original Bug": {
                "Error Type": record.error_type,
                "Original Error": record.buggy_output[:1000],  # Truncate for context
            },
            "Failed Debug Attempt": {
                "Attempt Number": trial.attempt_num,
                "Debug Plan": trial.debug_plan,
                "Code Executed": trial.code,
                "Error Output": trial.error_output,
                "Error Type": trial.error_type,
            },
            "Previous Failed Attempts": record.failed_strategies[-3:] if record.failed_strategies else [],
            "Task": (
                "Extract structured knowledge from this failure:\n\n"
                "1. WHY FAILED: Root cause of why THIS specific approach didn't work\n"
                "2. FAILED STRATEGY SUMMARY: One-line summary for RL\n"
                "3. LEARNED CONSTRAINT: Reusable rule to prevent repeating this\n\n"
                "This helps avoid retrying the same failed approach."
            ),
        }

        try:
            return query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_trial_failure_spec,
                model=self.model,
                temperature=0.1,
            )
        except Exception as e:
            logger.error("Failed to summarize trial failure: %s", e)
            # Fallback
            return {
                "why_failed": trial.error_output[:200] if trial.error_output else f"Failed with {trial.error_type}",
                "failed_strategy_summary": f"{trial.debug_plan[:100]} → Failed",
                "learned_constraint": f"Approach from attempt {trial.attempt_num} did not resolve {record.error_type}",
            }

    def _summarize_trial_success(self, trial: DebugTrial, record: BugRecord) -> dict:
        """
        Stage 2b: Summarize a successful debug trial for RL.
        Extracts why it worked, strategy summary, and key insight.
        """
        prompt = {
            "Your Role": "Analyze why this debug attempt succeeded",
            "Original Bug": {
                "Error Type": record.error_type,
                "Error Signature": record.error_signature or "Unknown",
                "Original Error": record.buggy_output[:1000],
            },
            "Successful Debug Attempt": {
                "Attempt Number": trial.attempt_num,
                "Debug Plan": trial.debug_plan,
                "Code Executed": trial.code[:2000],  # Show some code context
            },
            "Previous Failed Attempts": record.failed_strategies,
            "Task": (
                "Analyze WHY this approach succeeded where others failed.\n\n"
                "Extract:\n"
                "1. WHY WORKED: Specific technical reason this succeeded (1-2 sentences)\n"
                "2. SUCCESSFUL STRATEGY SUMMARY: One-line fix summary\n"
                "3. KEY INSIGHT: Core understanding\n\n"
                "Focus on the MECHANISM of success, not just stating 'it worked'."
            ),
        }

        try:
            result = query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_trial_success_spec,
                model=self.model,
                temperature=0.1,
            )
            # Validate that we got meaningful content
            why_worked = result.get("why_worked", "")
            if not why_worked or len(why_worked.strip()) < 20:
                raise ValueError("LLM returned empty or too-short why_worked")
            return result
        except Exception as e:
            logger.warning("Failed to summarize trial success for %s: %s. Using best-effort fallback.", record.bug_id, e)
            # Better fallback: extract from available context
            why_worked = trial.why_worked or "Code executed successfully"

            # Try to extract meaningful info from the debug plan
            plan_lines = trial.debug_plan.strip().split('\n') if trial.debug_plan else []
            strategy = plan_lines[0] if plan_lines else "Applied fix"

            return {
                "why_worked": why_worked,
                "successful_strategy_summary": strategy[:500],
                "key_insight": f"Approach resolved {record.error_type}",
            }

    def _summarize_bug_record(self, record: BugRecord) -> dict:
        prompt = {
            "Your Role": "Extract structured lessons from a debugging session",
            "Original Bug": {
                "Error Type": record.error_type,
                "Buggy Code": record.buggy_code,
                "Error Output": record.buggy_output,
                "Original Plan": record.original_plan,
            },
            "All Debug Trials": [
                {
                    "attempt": t.attempt_num,
                    "plan": t.debug_plan,
                    "outcome": t.outcome,
                    "error": t.error_type,
                    "why_worked": t.why_worked,
                    "why_failed": t.why_failed,
                }
                for t in record.trials
            ],
            "Final Outcome": record.final_outcome,
            "Task": (
                "Extract structured knowledge:\n\n"
                "1. ROOT CAUSE: What caused this bug?\n"
                "2. SUCCESSFUL STRATEGY: What worked? (if any)\n"
                "3. FAILED STRATEGIES: What didn't work?\n"
                "4. LESSON: One-line reusable pattern\n\n"
                "Be concise but complete. This becomes searchable knowledge."
            ),
        }

        try:
            return query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_bug_record_spec,
                model=self.model,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("Failed to summarize bug record: %s", e)
            sig = self._error_signature(record.buggy_output)
            return {
                "root_cause": (
                    f"{record.error_type}: {sig}"
                    if sig
                    else f"{record.error_type} (summarization failed)"
                ),
                "successful_strategy": (
                    record.trials[-1].debug_plan.strip().splitlines()[0][:200]
                    if record.trials
                    and record.trials[-1].outcome == "success"
                    and record.trials[-1].debug_plan
                    else None
                ),
                "failed_strategies": [
                    (
                        t.debug_plan.strip().splitlines()[0][:200]
                        if t.debug_plan
                        else f"Attempt {t.attempt_num}"
                    )
                    for t in record.trials
                    if t.outcome == "failed"
                ][:5],
                "lesson": (
                    f"For `{record.error_type}` with `{sig}`, apply the successful fix pattern from bug `{record.bug_id}`."
                    if sig
                    else f"For `{record.error_type}`, apply the successful fix pattern from bug `{record.bug_id}`."
                ),
            }

    # ═══════════════════════════════════════════════════════════
    # Persistence + world model writing
    # ═══════════════════════════════════════════════════════════

    def _prune_bug_records(self) -> None:
        if len(self.bug_records) <= self.max_bug_records:
            return
        sorted_records = sorted(self.bug_records.values(), key=lambda r: r.timestamp)
        to_remove = sorted_records[: len(self.bug_records) - self.max_bug_records]
        for r in to_remove:
            self.bug_records.pop(r.bug_id, None)
            if self.delete_pruned_bug_files and self.save_dir:
                try:
                    (self.save_dir / f"{r.bug_id}.md").unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

    def _enforce_active_bug_limit(self) -> None:
        """Safety valve: avoid unbounded growth of active bugs (rare; defaults are high)."""
        if self.max_active_bugs <= 0:
            return
        if len(self.active_bugs) < self.max_active_bugs:
            return
        # Evict oldest active bug record without an extra LLM summarization call.
        oldest_bug_id, _ = sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp)[0]
        self._auto_abandon_bug_record(oldest_bug_id, reason=f"Exceeded max_active_bugs={self.max_active_bugs}")

    def _auto_abandon_bug_record(self, bug_id: str, reason: str) -> None:
        """
        Mark an active bug record as abandoned due to memory limits.
        This avoids extra LLM calls (we record a concise lesson directly).
        """
        if bug_id not in self.active_bugs:
            return
        record = self.active_bugs[bug_id]
        record.final_outcome = "abandoned"
        if not record.root_cause:
            record.root_cause = f"Abandoned (memory safety valve): {reason}"
        if not record.lesson:
            record.lesson = (
                f"For `{record.error_type}`, repeated attempts did not resolve the issue; "
                f"abandon this path and try a different strategy/node. ({reason})"
            )
        self.bug_records[bug_id] = record
        self.active_bugs.pop(bug_id, None)
        self._prune_bug_records()
        self._save_bug_record(record, active=False)

    def _extract_error_signature_llm(self, output: str) -> tuple[str, str]:
        """
        Use LLM to extract a meaningful error signature and category from execution output.
        Returns: (error_signature, error_category)
        """
        if not output or len(output.strip()) < 10:
            return ("No error output", "UNKNOWN")

        try:
            result = query(
                system_message={
                    "Your Role": "Extract error signature from execution output",
                    "Execution Output": output,
                    "Task": (
                        "Extract the SPECIFIC error message that would help identify this exact bug.\n"
                        "Examples:\n"
                        "- 'TypeError: fit() got an unexpected keyword argument early_stopping_rounds'\n"
                        "- 'ValueError: Input contains NaN, infinity or a value too large for dtype(float64)'\n"
                        "- 'KeyError: \"['feature_x'] not found in axis\"'\n\n"
                        "Do NOT return generic messages like 'Traceback (most recent call last):'"
                    ),
                },
                user_message=None,
                func_spec=extract_error_signature_spec,
                model=self.model,
                temperature=0.0,
            )
            return (result.get("error_signature", "Unknown error"), result.get("error_category", "UNKNOWN"))
        except Exception as e:
            logger.debug("LLM error signature extraction failed: %s, falling back to heuristic", e)
            return (self._error_signature(output), "UNKNOWN")

    def _save_bug_record(self, record: BugRecord, active: bool = False) -> None:
        if not self.save_dir:
            return

        try:
            status_tag = "🔴 ACTIVE" if active else ("✅ COMPLETED" if record.final_outcome == "success" else "❌ FAILED")
            filepath = self.save_dir / f"{record.bug_id}.md"

            content: list[str] = [
                f"# {record.bug_id} {status_tag}",
                "",
                "## Bug Information",
                f"**Error Type**: {record.error_type}",
                f"**Error Signature**: {record.error_signature or '(not yet summarized)'}",
                f"**Error Category**: {record.error_category or 'Unknown'}",
                f"**Outcome**: {record.final_outcome}",
                f"**Trials**: {len(record.trials)}",
                "",
            ]

            # RAG fields (for semantic matching)
            if record.initial_hypothesis:
                content += ["## Initial Hypothesis", record.initial_hypothesis, ""]
            if record.context_tags:
                content += ["## Context Tags", f"`{', '.join(record.context_tags)}`", ""]

            # RL fields (what worked / what didn't)
            if record.successful_strategy:
                content += ["## Successful Strategy", record.successful_strategy, ""]
            if record.failed_strategies:
                content += ["## Failed Strategies", *[f"- {s}" for s in record.failed_strategies], ""]
            if record.learned_constraints:
                content += ["## Learned Constraints", *[f"- {c}" for c in record.learned_constraints], ""]

            # Completion fields
            if record.root_cause:
                content += ["## Root Cause", record.root_cause, ""]
            if record.lesson:
                content += ["## Lesson", record.lesson, ""]

            # Trial history
            content += ["## Debug Trials", ""]
            for trial in record.trials:
                content += [
                    f"### Attempt {trial.attempt_num}: {trial.outcome.upper()}",
                    "",
                    "**Plan**:",
                    "```",
                    trial.debug_plan or "No plan",
                    "```",
                    "",
                ]
                if trial.outcome == "failed":
                    content.append(f"**Why Failed**: {trial.why_failed or 'Not recorded'}")
                    if trial.error_type:
                        content.append(f"**Error Type**: {trial.error_type}")
                else:
                    content.append(f"**Why Worked**: {trial.why_worked or 'Not recorded'}")
                content += ["", "---", ""]

            filepath.write_text("\n".join(content))
        except Exception as e:
            logger.error("Failed to save bug record: %s", e)

    @staticmethod
    def _error_signature(output: str | None) -> str:
        """
        Extract a compact, searchable error signature from a traceback/log.
        Prefers the final exception line when present.
        """
        if not output:
            return ""
        text = str(output).strip()
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Heuristic: prefer lines that look like an exception summary (even if the log is truncated).
        candidates: list[str] = []
        for ln in lines:
            if "traceback" in ln.lower():
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b", ln) and ":" in ln:
                candidates.append(ln)
        if candidates:
            return candidates[-1][:300]

        # Prefer the final "XError: message" line.
        for ln in reversed(lines[-50:]):
            if "state:" in ln.lower():
                continue
            # Match "TypeError: ...", "ValueError: ...", "Exception: ...", etc.
            # (Avoid word-boundary pitfalls like "TypeError" where "Error" isn't a separate word.)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b", ln) and ":" in ln:
                return ln[:300]

        # Secondary: find any exception-ish line even without a colon (rare).
        for ln in reversed(lines[-50:]):
            if re.search(r"(Error|Exception)\b", ln) and "traceback" not in ln.lower():
                return ln[:300]

        return lines[0][:300] if lines else ""

    def format_active_trial_history(self, bug_id: str, *, max_trials: int = 5) -> str:
        """
        Deterministic, no-LLM summary of active attempts on the current bug.
        Used to prevent repeating known-bad strategies during multi-attempt debugging.
        """
        record = self.active_bugs.get(bug_id)
        if record is None or not record.trials:
            return ""

        trials = record.trials[-max(1, int(max_trials)) :]
        out: list[str] = []
        out.append(f"Bug `{bug_id}` active attempts (most recent last):")
        for t in trials:
            plan_line = (t.debug_plan or "").strip().splitlines()[0:1]
            plan_short = plan_line[0].strip() if plan_line else "(no plan)"
            if t.outcome == "failed":
                why = (t.why_failed or t.error_output or t.error_type or "").strip()
                why = " ".join(why.split())[:240]
                out.append(f"- Attempt {t.attempt_num}: FAILED — {plan_short}")
                if why:
                    out.append(f"  - Why failed: {why}")
            else:
                why = (t.why_worked or "").strip()
                why = " ".join(why.split())[:240]
                out.append(f"- Attempt {t.attempt_num}: SUCCESS — {plan_short}")
                if why:
                    out.append(f"  - Why worked: {why}")
        return "\n".join(out).strip()

    def _render_world_model(self, journal: Optional["Journal"] = None) -> str:
        lines: list[str] = []
        lines.append(f"# World Model (Semantic Policy) — v{self.world_model_version}")
        lines.append("")

        # 1) Operator priors (derived from journal if provided)
        lines.append("## 1) Operator Priors (Observed)")
        if journal is None:
            lines.append("- (journal not provided)")
        else:
            nodes = journal.nodes
            total = len(nodes)
            buggy = sum(1 for n in nodes if n.is_buggy)
            ok = total - buggy

            def _count(stage: str, ok_only: bool) -> int:
                if ok_only:
                    return sum(1 for n in nodes if n.stage_name == stage and not n.is_buggy)
                return sum(1 for n in nodes if n.stage_name == stage)

            for stage in ("draft", "improve", "debug"):
                lines.append(
                    f"- `{stage}`: { _count(stage, False) } runs, { _count(stage, True) } valid"
                )
            lines.append(f"- Total: {total} runs, {ok} valid, {buggy} buggy")
        lines.append("")

        # 2/3) What worked / failed from bug records
        lines.append("## 2) What Worked (Evidence-Based)")
        worked = [
            r for r in self.bug_records.values() if r.final_outcome == "success" and r.successful_strategy
        ]
        if not worked:
            lines.append("- (no completed successful fixes yet)")
        else:
            for r in sorted(worked, key=lambda x: x.timestamp, reverse=True)[:15]:
                lines.append(
                    f"- IF `{r.error_type}` THEN `{r.successful_strategy}` (bug_id={r.bug_id}, trials={len(r.trials)})"
                )
        lines.append("")

        lines.append("## 3) What Failed (Evidence-Based)")
        failed_items: list[str] = []
        for r in sorted(self.bug_records.values(), key=lambda x: x.timestamp, reverse=True)[:30]:
            for s in r.failed_strategies or []:
                failed_items.append(f"- IF `{r.error_type}` THEN `{s}` fails (bug_id={r.bug_id})")
        if not failed_items:
            lines.append("- (no recorded failed strategies yet)")
        else:
            lines.extend(failed_items[:20])
        lines.append("")

        lines.append("## 4) Lessons (Compact)")
        lessons = [
            (r.error_type, r.lesson, r.bug_id)
            for r in sorted(self.bug_records.values(), key=lambda x: x.timestamp, reverse=True)
            if r.lesson
        ]
        if not lessons:
            lines.append("- (no lessons yet)")
        else:
            for et, lesson, bug_id in lessons[:20]:
                lines.append(f"- `{et}`: {lesson} ({bug_id})")
        lines.append("")

        if self.active_bugs:
            lines.append("## Active Bugs (In Progress)")
            for bug_id, r in sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp, reverse=True)[:10]:
                lines.append(
                    f"- {bug_id}: `{r.error_type}` (trials so far: {len(r.trials)})"
                )
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _render_bug_index(self) -> str:
        """
        Render a table of contents / index of all bugs for RAG.
        This allows quick scanning and semantic matching of bugs.
        """
        lines: list[str] = []
        lines.append("# Bug Index / Table of Contents")
        lines.append("")
        lines.append("Quick reference for all bugs encountered. Use this for RAG to find relevant bugs.")
        lines.append("")

        # Summary stats
        total = len(self.bug_records) + len(self.active_bugs)
        completed = len([r for r in self.bug_records.values() if r.final_outcome == "success"])
        abandoned = len([r for r in self.bug_records.values() if r.final_outcome == "abandoned"])
        active = len(self.active_bugs)

        lines.append(f"**Total Bugs**: {total} | **Completed**: {completed} | **Abandoned**: {abandoned} | **Active**: {active}")
        lines.append("")

        # Table header
        lines.append("| Bug ID | Status | Error Type | Error Signature | Category | Trials | Outcome |")
        lines.append("|--------|--------|------------|-----------------|----------|--------|---------|")

        # Active bugs first
        for bug_id, record in sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp, reverse=True):
            sig = record.error_signature or self._error_signature(record.buggy_output) or "Unknown"
            sig = sig[:80] + "..." if len(sig) > 80 else sig
            category = record.error_category or "Unknown"
            lines.append(
                f"| {bug_id} | 🔴 ACTIVE | {record.error_type} | {sig} | {category} | {len(record.trials)} | in_progress |"
            )

        # Completed bugs
        for bug_id, record in sorted(self.bug_records.items(), key=lambda t: t[1].timestamp, reverse=True):
            sig = record.error_signature or self._error_signature(record.buggy_output) or "Unknown"
            sig = sig[:80] + "..." if len(sig) > 80 else sig
            category = record.error_category or "Unknown"
            status = "✅" if record.final_outcome == "success" else "❌"
            lines.append(
                f"| {bug_id} | {status} | {record.error_type} | {sig} | {category} | {len(record.trials)} | {record.final_outcome} |"
            )

        lines.append("")
        lines.append("## Quick Access by Category")
        lines.append("")

        # Group by category
        by_category: dict[str, list[str]] = {}
        for bug_id, record in list(self.active_bugs.items()) + list(self.bug_records.items()):
            cat = record.error_category or "Unknown"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(bug_id)

        for category in sorted(by_category.keys()):
            bugs = by_category[category]
            lines.append(f"- **{category}**: {', '.join(sorted(bugs))}")

        lines.append("")
        lines.append("## What Worked (Quick Reference)")
        lines.append("")
        for record in sorted(self.bug_records.values(), key=lambda r: r.timestamp, reverse=True):
            if record.final_outcome == "success" and record.successful_strategy:
                lines.append(f"- **{record.bug_id}** ({record.error_type}): {record.successful_strategy}")

        lines.append("")
        lines.append("## What Failed (Quick Reference)")
        lines.append("")
        for record in sorted(self.bug_records.values(), key=lambda r: r.timestamp, reverse=True)[:10]:
            if record.failed_strategies:
                lines.append(f"- **{record.bug_id}** ({record.error_type}):")
                for strategy in record.failed_strategies[:3]:
                    lines.append(f"  - {strategy}")

        return "\n".join(lines) + "\n"

    def _save_bug_index(self) -> None:
        """Save the bug index for quick RAG access."""
        if not self.save_dir:
            return
        try:
            content = self._render_bug_index()
            path = self.save_dir / "BUG_INDEX.md"
            path.write_text(content)
        except Exception as e:
            logger.error("Failed to save bug index: %s", e)

    def _save_world_model(self, journal: Optional["Journal"] = None) -> None:
        if not self.save_dir:
            return
        try:
            self.world_model_version += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            content = self._render_world_model(journal=journal)
            path = self.save_dir / "world_model_LATEST.md"
            path.write_text(f"<!-- updated: {ts} -->\n\n{content}")
            # Also update bug index
            self._save_bug_index()
        except Exception as e:
            logger.error("Failed to save world model: %s", e)
