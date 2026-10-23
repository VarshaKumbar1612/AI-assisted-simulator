"""
Session-state case management for the investigation simulator.

Streamlit reruns the script on every click. This module is the single place
that owns ``st.session_state`` for the case:

* :func:`initialize_case` loads every file once via ``load_all_evidence()``.
* :func:`visible_evidence` is the visibility filter (``access == "public"``).
* :func:`unlock_record` flips a hidden file to public and stores the
  investigator's justification.

An optional ``state`` mapping can be passed instead of Streamlit so the same
rules can be unit-tested without a browser session.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any, MutableMapping

import streamlit as st

from document_processor import load_all_evidence

CASE_LOADED_KEY = "case_loaded"
EVIDENCE_KEY = "evidence"
UNLOCK_LOG_KEY = "unlock_log"
CHAT_KEY = "chat_messages"
DECISION_HISTORY_KEY = "decision_history"
CURRENT_DECISION_KEY = "current_decision"
DECISION_REVISION_KEY = "decision_revision"

State = MutableMapping[str, Any]


def _bucket(state: State | None) -> State:
    """Return the Streamlit session, or a caller-supplied dict for tests."""
    return state if state is not None else st.session_state


def initialize_case(state: State | None = None) -> None:
    """
    Load the full case file into session state if it is not already there.

    Hidden records are stored with ``access='hidden'`` so the UI and the LLM
    cannot see their text until :func:`unlock_record` is called.
    """
    bucket = _bucket(state)
    if bucket.get(CASE_LOADED_KEY):
        return

    evidence = copy.deepcopy(load_all_evidence())
    for record in evidence.values():
        record["originally_hidden"] = record.get("access") == "hidden"

    bucket[EVIDENCE_KEY] = evidence
    bucket[UNLOCK_LOG_KEY] = []
    bucket[CHAT_KEY] = []
    bucket[DECISION_HISTORY_KEY] = []
    bucket[CURRENT_DECISION_KEY] = None
    bucket[DECISION_REVISION_KEY] = 1
    bucket[CASE_LOADED_KEY] = True


def get_evidence(state: State | None = None) -> dict[str, dict[str, Any]]:
    """Return the live evidence dictionary from session state."""
    initialize_case(state)
    return _bucket(state)[EVIDENCE_KEY]


def visible_evidence(state: State | None = None) -> dict[str, dict[str, Any]]:
    """
    Visibility filter: records the frontend (and the LLM) may currently show.

    Only items whose ``access`` field is ``public`` are returned. Originally
    public files start here; unlocked files join after :func:`unlock_record`.
    """
    return {
        evidence_id: record
        for evidence_id, record in get_evidence(state).items()
        if record.get("access") == "public"
    }


def hidden_evidence(state: State | None = None) -> dict[str, dict[str, Any]]:
    """Records that still require a justification to unlock."""
    return {
        evidence_id: record
        for evidence_id, record in get_evidence(state).items()
        if record.get("access") == "hidden"
    }


def unlocked_evidence_ids(state: State | None = None) -> list[str]:
    """Evidence IDs currently in the investigator's hand (all public records)."""
    return list(visible_evidence(state).keys())


def unlock_log(state: State | None = None) -> list[dict[str, Any]]:
    """Chronological list of successful unlocks and their justifications."""
    initialize_case(state)
    return list(_bucket(state).get(UNLOCK_LOG_KEY, []))


def unlock_record(
    evidence_id: str,
    justification: str,
    state: State | None = None,
) -> tuple[bool, str]:
    """
    Promote a hidden document to public and log the user's justification.

    Returns ``(True, message)`` on success, or ``(False, error)`` if the ID is
    missing, already public, or the justification is empty.
    """
    initialize_case(state)
    bucket = _bucket(state)
    evidence_id = (evidence_id or "").strip()
    reason = (justification or "").strip()

    if not evidence_id:
        return False, "Select a restricted record to request."
    if len(reason) < 10:
        return False, "Write a justification of at least 10 characters."

    evidence = bucket[EVIDENCE_KEY]
    record = evidence.get(evidence_id)
    if record is None:
        return False, f"{evidence_id} is not in the case file."
    if record.get("access") == "public":
        return False, f"{evidence_id} is already in your evidence board."
    if record.get("access") != "hidden":
        return False, f"{evidence_id} cannot be unlocked from its current status."

    record["access"] = "public"
    entry = {
        "evidence_id": evidence_id,
        "source_file": record.get("source_file", ""),
        "justification": reason,
        "unlocked_at": datetime.now(timezone.utc).isoformat(),
    }
    log = list(bucket.get(UNLOCK_LOG_KEY, []))
    log.append(entry)
    # Reassign so Streamlit notices the mutation on nested dicts/lists.
    bucket[EVIDENCE_KEY] = evidence
    bucket[UNLOCK_LOG_KEY] = log
    return True, f"{evidence_id} unlocked and added to the evidence board."


def public_evidence_corpus(state: State | None = None) -> str:
    """
    Concatenate currently public records for the interrogation LLM.

    Hidden text is never included, even though it sits in session state.
    """
    blocks: list[str] = []
    for evidence_id, record in visible_evidence(state).items():
        source = record.get("source_file", "")
        text = (record.get("text") or "").strip()
        blocks.append(
            f"===== {evidence_id} | {source} =====\n{text}\n===== END {evidence_id} ====="
        )
    return "\n\n".join(blocks)


def append_chat_turn(
    role: str,
    content: str,
    state: State | None = None,
    **extra: Any,
) -> None:
    """Append one chat message to the interrogation history."""
    initialize_case(state)
    bucket = _bucket(state)
    history = list(bucket.get(CHAT_KEY, []))
    turn: dict[str, Any] = {"role": role, "content": content}
    turn.update(extra)
    history.append(turn)
    bucket[CHAT_KEY] = history


def chat_history(state: State | None = None) -> list[dict[str, Any]]:
    """Return interrogation messages for the chat renderer."""
    initialize_case(state)
    return list(_bucket(state).get(CHAT_KEY, []))


def detect_evidence_conflicts(
    state: State | None = None,
) -> list[dict[str, Any]]:
    """Compare quantities, named dates, and opposing directives in public files."""
    observations: dict[str, dict[str, set[int]]] = {}
    excerpt_by_id: dict[str, str] = {}
    date_observations: dict[str, dict[str, set[str]]] = {}
    directive_observations: dict[str, dict[str, tuple[str, str]]] = {}

    for evidence_id, record in visible_evidence(state).items():
        source_name = str(record.get("source_file", "")).lower()
        text = str(record.get("text") or "")
        excerpt_by_id[evidence_id] = str(record.get("source_file", evidence_id))
        categories: dict[str, set[int]] = {"intended": set(), "received": set()}
        dates: dict[str, set[str]] = {}

        for line in text.splitlines():
            lowered = line.lower()
            if not re.search(r"\b(shortage|shortfall|missing)\b", lowered):
                quantity_match = re.search(
                    r"\b([\d,]+)\s*(?:pieces?|pcs?)\b", line, flags=re.IGNORECASE
                )
                if not quantity_match:
                    quantity_match = re.search(
                        r"\b(?:pieces?|pcs?)\s+"
                        r"(?:counted|received|invoiced|ordered|shipped)\s*[:=-]?\s*"
                        r"([\d,]+)\b",
                        line,
                        flags=re.IGNORECASE,
                    )
                if quantity_match:
                    quantity = int(quantity_match.group(1).replace(",", ""))
                    if re.search(r"\b(counted|received|physical count)\b", lowered):
                        categories["received"].add(quantity)
                    if re.search(
                        r"\b(invoic\w*|order\w*|dispatch\w*|shipment)\b", lowered
                    ) or (
                        "quantity" in lowered
                        and ("invoice" in source_name or "purchase_order" in source_name)
                    ):
                        categories["intended"].add(quantity)

            date_labels = (
                ("dispatch_date", r"\bdispatch date\b"),
                ("invoice_date", r"\binvoice date\b"),
                ("po_date", r"\bpo date\b|\bpurchase order date\b"),
                ("receipt_date", r"\bgrn date\b|\breceipt date\b|\breceived date\b"),
                ("delivery_date", r"\brequested delivery\b|\bdelivery date\b"),
            )
            for label, pattern in date_labels:
                if re.search(pattern, lowered):
                    found_dates = re.findall(
                        r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", line
                    )
                    if found_dates:
                        dates.setdefault(label, set()).update(
                            re.sub(r"\s+", " ", item).strip().lower()
                            for item in found_dates
                        )

        observations[evidence_id] = categories
        date_observations[evidence_id] = dates
        normalized = re.sub(r"\s+", " ", text)
        directives: dict[str, tuple[str, str]] = {}
        directive_patterns = {
            "dashboard": (
                ("prohibited", r"do not [^.]{0,120}(?:dashboard|publish|escalate)[^.]*"),
                ("required", r"[^.]{0,140}must [^.]{0,120}dashboard[^.]*"),
            ),
            "internal_audit": (
                ("prohibited", r"do not copy [^.]{0,100}internal audit[^.]*"),
                ("required", r"[^.]{0,140}(?:must|shall) [^.]{0,120}internal audit[^.]*"),
            ),
        }
        for topic, patterns in directive_patterns.items():
            for status, pattern in patterns:
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match:
                    directives[topic] = (status, match.group(0).strip())
                    break
        directive_observations[evidence_id] = directives

    conflicts: list[dict[str, Any]] = []
    evidence_ids = list(observations)
    for intended_id in evidence_ids:
        intended_values = observations[intended_id]["intended"]
        if not intended_values:
            continue
        for received_id in evidence_ids:
            received_values = observations[received_id]["received"]
            if intended_id == received_id or not received_values:
                continue
            for intended in sorted(intended_values):
                for received in sorted(received_values):
                    if intended == received:
                        continue
                    conflicts.append(
                        {
                            "type": "quantity_mismatch",
                            "intended_evidence_id": intended_id,
                            "intended_source": excerpt_by_id[intended_id],
                            "intended_quantity": intended,
                            "received_evidence_id": received_id,
                            "received_source": excerpt_by_id[received_id],
                            "received_quantity": received,
                        }
                    )

    for first_index, first_id in enumerate(evidence_ids):
        for second_id in evidence_ids[first_index + 1 :]:
            for label in date_observations[first_id].keys() & date_observations[second_id].keys():
                first_dates = date_observations[first_id][label]
                second_dates = date_observations[second_id][label]
                for first_date in sorted(first_dates):
                    for second_date in sorted(second_dates):
                        if first_date != second_date:
                            conflicts.append(
                                {
                                    "type": "date_mismatch",
                                    "date_label": label.replace("_", " "),
                                    "first_evidence_id": first_id,
                                    "first_source": excerpt_by_id[first_id],
                                    "first_date": first_date,
                                    "second_evidence_id": second_id,
                                    "second_source": excerpt_by_id[second_id],
                                    "second_date": second_date,
                                }
                            )
            for topic in directive_observations[first_id].keys() & directive_observations[second_id].keys():
                first_status, first_claim = directive_observations[first_id][topic]
                second_status, second_claim = directive_observations[second_id][topic]
                if first_status != second_status:
                    conflicts.append(
                        {
                            "type": "claim_mismatch",
                            "topic": topic.replace("_", " "),
                            "first_evidence_id": first_id,
                            "first_source": excerpt_by_id[first_id],
                            "first_claim": first_claim,
                            "second_evidence_id": second_id,
                            "second_source": excerpt_by_id[second_id],
                            "second_claim": second_claim,
                        }
                    )
    return conflicts


def generate_final_report(
    conflicts: list[dict[str, Any]] | None = None,
    state: State | None = None,
) -> str:
    """Build a Markdown report from unlocked evidence and decision history."""
    initialize_case(state)
    bucket = _bucket(state)
    case_conflicts = conflicts if conflicts is not None else detect_evidence_conflicts(state)
    attempts = bucket.get(DECISION_HISTORY_KEY, [])
    lines = [
        "# Apex Electronics Investigation Report",
        "",
        "## Unlocked evidence timeline",
    ]
    log = unlock_log(state)
    if not log:
        lines.append("- No restricted evidence was unlocked.")
    for entry in log:
        lines.append(
            f"- {entry.get('unlocked_at', 'Time not recorded')}: "
            f"{entry.get('evidence_id')} ({entry.get('source_file', 'Unknown file')}) — "
            f"{entry.get('justification', '')}"
        )

    lines.extend(["", "## Unresolved conflicts"])
    if not case_conflicts:
        lines.append("- No cross-document quantity conflicts detected.")
    for conflict in case_conflicts:
        if conflict["type"] == "quantity_mismatch":
            lines.append(
                f"- Intended/invoiced **{conflict['intended_quantity']} pieces** "
                f"({conflict['intended_evidence_id']}, {conflict['intended_source']}) "
                f"differs from received/count **{conflict['received_quantity']} pieces** "
                f"({conflict['received_evidence_id']}, {conflict['received_source']})."
            )
        elif conflict["type"] == "date_mismatch":
            lines.append(
                f"- Conflicting {conflict['date_label']}: **{conflict['first_date']}** "
                f"({conflict['first_evidence_id']}, {conflict['first_source']}) versus "
                f"**{conflict['second_date']}** ({conflict['second_evidence_id']}, "
                f"{conflict['second_source']})."
            )
        else:
            lines.append(
                f"- Conflicting {conflict['topic']} instructions in "
                f"{conflict['first_evidence_id']} ({conflict['first_source']}) and "
                f"{conflict['second_evidence_id']} ({conflict['second_source']}): "
                f"{conflict['first_claim']} / {conflict['second_claim']}."
            )

    lines.extend(["", "## Decision history"])
    if not attempts:
        lines.append("- No decisions have been submitted.")
    for index, attempt in enumerate(attempts, start=1):
        submission = attempt.get("submission", {})
        evaluation = attempt.get("evaluation", {})
        lines.extend(
            [
                f"### Attempt {index} — {attempt.get('submitted_at', 'Time not recorded')}",
                f"- Score: {evaluation.get('score', 'Not available')}/100",
                f"- Conclusion: {submission.get('conclusion', '')}",
                f"- Supporting evidence: {submission.get('supporting_evidence', '')}",
                f"- Counterevidence: {submission.get('counterevidence', '')}",
                f"- Confidence: {submission.get('confidence', '')}%",
                f"- Recommended next action: {submission.get('next_action', '')}",
            ]
        )
        for criterion in evaluation.get("criteria", []):
            lines.append(
                f"- {criterion.get('name', 'Criterion')}: "
                f"{criterion.get('score', '—')}/20 — {criterion.get('feedback', '')}"
            )

    lines.extend(["", "## Final recommended steps"])
    if attempts:
        final_submission = attempts[-1].get("submission", {})
        lines.append(final_submission.get("next_action") or "- No recommendation provided.")
    else:
        lines.append("- No final recommendation submitted.")
    lines.append("")
    return "\n".join(lines)
