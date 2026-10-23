"""
Streamlit frontend for the AI-assisted investigation simulator.

Tabs
----
* Evidence Board — currently public records in expanders.
* Interrogation — chat against unlocked text only.
* Request Records — justify unlocking a hidden file.
* Final Decision — submit, revise, score, and export the case report.

Case data is loaded once into ``st.session_state`` via
:func:`investigator.initialize_case`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

import streamlit as st

from investigator import (
    CURRENT_DECISION_KEY,
    DECISION_HISTORY_KEY,
    DECISION_REVISION_KEY,
    append_chat_turn,
    chat_history,
    detect_evidence_conflicts,
    generate_final_report,
    hidden_evidence,
    initialize_case,
    public_evidence_corpus,
    unlock_log,
    unlock_record,
    unlocked_evidence_ids,
    visible_evidence,
)
from llm import (
    LLMConfigurationError,
    LLMJSONError,
    api_configured,
    format_interrogation_reply,
    interrogate,
    score_decision,
)

st.set_page_config(
    page_title="Apex Electronics — Missing Components",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_case()


def render_sidebar() -> None:
    """Scenario briefing plus the live list of Evidence IDs in hand."""
    st.sidebar.title("Case file")
    st.sidebar.subheader("Apex Electronics — ₹18 lakh shipment")
    st.sidebar.markdown(
        """
Apex Electronics paid **INR 18,00,000** for **5,000** NEX-4407 MOSFETs
from NovaSemi Components. Warehouse B in Pune logged a **shortfall of
500 pieces**. Procurement asked staff not to escalate the gap.

Start with the public pack. Restricted files stay sealed until you
justify a records request.
        """.strip()
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Evidence IDs in hand**")
    in_hand = unlocked_evidence_ids()
    if in_hand:
        for evidence_id in in_hand:
            record = visible_evidence()[evidence_id]
            badge = "unlocked" if record.get("originally_hidden") else "public"
            st.sidebar.write(f"`{evidence_id}` · {badge} · {record['source_file']}")
    else:
        st.sidebar.caption("No records are currently visible.")

    still_hidden = hidden_evidence()
    st.sidebar.caption(
        f"{len(still_hidden)} restricted file(s) remain sealed."
        if still_hidden
        else "All restricted files have been unlocked."
    )


def render_evidence_board() -> None:
    """Show raw text of every currently public document."""
    st.markdown("Review the files you are allowed to see. Restricted records are not listed here.")
    conflicts = detect_evidence_conflicts()
    if conflicts:
        conflict_lines = []
        for conflict in conflicts:
            if conflict["type"] == "quantity_mismatch":
                description = (
                    f"<strong>{conflict['intended_evidence_id']}</strong> "
                    f"({escape(conflict['intended_source'])}) indicates "
                    f"{conflict['intended_quantity']:,} pieces, while "
                    f"<strong>{conflict['received_evidence_id']}</strong> "
                    f"({escape(conflict['received_source'])}) records "
                    f"{conflict['received_quantity']:,} received/count."
                )
            elif conflict["type"] == "date_mismatch":
                description = (
                    f"{escape(conflict['date_label'])} differs between "
                    f"<strong>{conflict['first_evidence_id']}</strong> "
                    f"({escape(conflict['first_source'])}: {escape(conflict['first_date'])}) "
                    f"and <strong>{conflict['second_evidence_id']}</strong> "
                    f"({escape(conflict['second_source'])}: {escape(conflict['second_date'])})."
                )
            else:
                description = (
                    f"{escape(conflict['topic'])} instructions conflict between "
                    f"<strong>{conflict['first_evidence_id']}</strong> "
                    f"({escape(conflict['first_source'])}: {escape(conflict['first_claim'])}) "
                    f"and <strong>{conflict['second_evidence_id']}</strong> "
                    f"({escape(conflict['second_source'])}: {escape(conflict['second_claim'])})."
                )
            conflict_lines.append(f"<li>{description}</li>")
        st.markdown(
            '<div style="background-color:#4a2600;border-left:6px solid #ff8c00;'
            'padding:1rem;border-radius:.4rem;color:#fff3e0">'
            "<strong>Quantity conflict detected</strong><ul>"
            f"{''.join(conflict_lines)}</ul></div>",
            unsafe_allow_html=True,
        )
    public = visible_evidence()
    if not public:
        st.warning("No public evidence is loaded.")
        return
    for evidence_id, record in public.items():
        title = f"{evidence_id} — {record['source_file']}"
        with st.expander(title, expanded=False):
            if record.get("originally_hidden"):
                st.caption("Unlocked from restricted storage.")
            st.text(record.get("text") or "")


def render_interrogation() -> None:
    """Chat UI that sends questions plus currently public text to Gemini."""
    st.markdown(
        "Ask questions about the case. The assistant sees **only** the documents "
        "on your evidence board. Hidden files are invisible to it."
    )
    if not api_configured():
        st.error("Set `GEMINI_API_KEY` in `.env` before running interrogation.")

    prompt = st.chat_input("Ask a question based on the unlocked records…")
    if prompt:
        append_chat_turn("user", prompt)

    for turn in chat_history():
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if not prompt:
        return

    corpus = public_evidence_corpus()
    with st.chat_message("assistant"):
        if not api_configured():
            reply = "The records do not contain this information"
            st.markdown(reply)
            append_chat_turn("assistant", reply, found_in_records=False)
            return
        try:
            with st.spinner("Checking unlocked records…"):
                payload = interrogate(prompt, corpus)
            reply = format_interrogation_reply(payload)
        except (LLMConfigurationError, LLMJSONError, ValueError) as exc:
            reply = f"The assistant could not complete that check: {exc}"
            payload = {}
        except Exception as exc:  # network / provider errors
            reply = f"The assistant could not complete that check: {exc}"
            payload = {}
        st.markdown(reply)
        append_chat_turn(
            "assistant",
            reply,
            found_in_records=bool(payload.get("found_in_records")),
            cited_evidence_ids=payload.get("cited_evidence_ids") or [],
        )


def render_request_records() -> None:
    """Unlock form: request a hidden Evidence ID and record the justification."""
    st.markdown(
        "Restricted files are not shown on the evidence board and are not sent "
        "to the interrogation assistant until you unlock them."
    )
    if not hidden_evidence():
        st.success("There are no remaining restricted records.")
    else:
        with st.form("unlock_form", clear_on_submit=True):
            evidence_id = st.text_input(
                "Evidence ID",
                placeholder="For example, EVID-005",
            )
            justification = st.text_area(
                "Justification",
                placeholder="Why do the public records make this file necessary?",
                height=120,
            )
            submitted = st.form_submit_button("Unlock record")
        if submitted:
            ok, message = unlock_record(evidence_id, justification)
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    log = unlock_log()
    if log:
        st.divider()
        st.markdown("**Unlock log**")
        for entry in log:
            st.write(
                f"`{entry['evidence_id']}` — {entry['source_file']}\n\n"
                f"{entry['justification']}"
            )


def render_final_decision() -> None:
    """Collect, score, and retain decision attempts; provide an exportable report."""
    st.markdown(
        "Submit a conclusion grounded in the records currently on your evidence board. "
        "Each criterion is graded against the same 100-point rubric."
    )
    current = st.session_state.get(CURRENT_DECISION_KEY)
    if current:
        evaluation = current["evaluation"]
        st.metric("Decision score", f"{evaluation['score']}/100")
        st.markdown(evaluation["overall_feedback"])
        for criterion in evaluation["criteria"]:
            st.markdown(
                f"**{criterion['name']} — {criterion['score']}/20**  \n"
                f"{criterion['feedback']}"
            )
        if evaluation.get("missed_issues"):
            st.markdown("**Missed issues**")
            for issue in evaluation["missed_issues"]:
                st.write(f"- {issue}")
        if evaluation.get("unsupported_claims"):
            st.markdown("**Unsupported claims**")
            for claim in evaluation["unsupported_claims"]:
                st.write(f"- {claim}")
        if st.button("Revise Decision", key="revise_decision"):
            st.session_state[DECISION_REVISION_KEY] += 1
            st.session_state[CURRENT_DECISION_KEY] = None
            st.rerun()

    revision = st.session_state[DECISION_REVISION_KEY]
    with st.form(f"decision_form_{revision}"):
        conclusion = st.text_area(
            "Final conclusion",
            key=f"decision_conclusion_{revision}",
            placeholder="What happened to the missing shipment quantity, and who is responsible?",
            height=130,
        )
        supporting_evidence = st.text_area(
            "Supporting evidence (include Evidence IDs)",
            key=f"decision_supporting_{revision}",
            placeholder="Identify the specific records and facts supporting your conclusion.",
            height=110,
        )
        counterevidence = st.text_area(
            "Counterevidence and alternative explanations",
            key=f"decision_counterevidence_{revision}",
            placeholder="What evidence could challenge your conclusion?",
            height=100,
        )
        confidence = st.slider(
            "Confidence level",
            min_value=0,
            max_value=100,
            value=50,
            key=f"decision_confidence_{revision}",
        )
        next_action = st.text_area(
            "Recommended next action",
            key=f"decision_next_action_{revision}",
            placeholder="What should the organization do next?",
            height=100,
        )
        submitted = st.form_submit_button("Submit decision for grading")

    if submitted:
        if not conclusion.strip() or not supporting_evidence.strip() or not next_action.strip():
            st.error("Complete your conclusion, supporting evidence, and recommended next action.")
        elif not api_configured():
            st.error("Grading is unavailable. Configure `GEMINI_API_KEY`, then submit again.")
        else:
            submission = {
                "conclusion": conclusion.strip(),
                "supporting_evidence": supporting_evidence.strip(),
                "counterevidence": counterevidence.strip() or "None identified.",
                "confidence": confidence,
                "next_action": next_action.strip(),
            }
            try:
                with st.spinner("Evaluating your decision against the case records…"):
                    evaluation = score_decision(
                        submission,
                        available_evidence=public_evidence_corpus(),
                        conflicts=detect_evidence_conflicts(),
                    )
            except (LLMConfigurationError, LLMJSONError, ValueError) as exc:
                st.error(
                    "We couldn't grade this submission. Please check it and submit again."
                )
                st.caption(f"Details: {exc}")
            except Exception as exc:
                st.error(
                    "The grading service is temporarily unavailable. Please submit again."
                )
                st.caption(f"Details: {exc}")
            else:
                attempt = {
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "submission": submission,
                    "evaluation": evaluation,
                }
                history = list(st.session_state[DECISION_HISTORY_KEY])
                history.append(attempt)
                st.session_state[DECISION_HISTORY_KEY] = history
                st.session_state[CURRENT_DECISION_KEY] = attempt
                st.rerun()

    history = st.session_state[DECISION_HISTORY_KEY]
    if history:
        st.subheader("Decision attempt history")
        for index, attempt in enumerate(history, start=1):
            with st.expander(
                f"Attempt {index} — {attempt['evaluation']['score']}/100"
            ):
                st.write(attempt["submission"]["conclusion"])
                st.caption(attempt["evaluation"]["overall_feedback"])

    st.divider()
    final_report_string = generate_final_report(detect_evidence_conflicts())
    st.download_button(
        label="Download Final Report",
        data=final_report_string,
        file_name="case_resolution_report.txt",
        mime="text/plain",
    )


render_sidebar()
st.title("Investigation workspace")
st.caption("Apex Electronics · missing 500 electronic components · INR 18,00,000 shipment")

tab_board, tab_chat, tab_request, tab_decision = st.tabs(
    ["Evidence Board", "Interrogation", "Request Records", "Final Decision"]
)
with tab_board:
    render_evidence_board()
with tab_chat:
    render_interrogation()
with tab_request:
    render_request_records()
with tab_decision:
    render_final_decision()
