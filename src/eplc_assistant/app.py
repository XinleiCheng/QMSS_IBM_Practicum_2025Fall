"""Streamlit interface for grounded EPLC assistance."""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from eplc_assistant.client import AssistantApiClient, ApiClientError
from eplc_assistant.models import Citation
from eplc_assistant.templates import PHASE_TEMPLATES


def main() -> None:
    st.set_page_config(
        page_title="EPLC Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_state()
    _render_sidebar()

    page = st.session_state.page
    if page == "Learn":
        _render_learn_page()
    elif page == "Ask":
        _render_qna_page()
    else:
        _render_drafting_page()


def _initialize_state() -> None:
    st.session_state.setdefault("page", "Learn")
    st.session_state.setdefault("messages", [])


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("EPLC Assistant")
        st.caption(
            "Grounded guidance and drafting support for IT project managers."
        )
        st.divider()
        page = st.radio(
            "Navigation",
            ("Learn", "Ask", "Create"),
            index=("Learn", "Ask", "Create").index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.session_state.page = page
        st.divider()
        st.caption(
            "AI output is a drafting aid, not an official compliance determination."
        )


def _render_learn_page() -> None:
    st.title("Work with EPLC guidance, one decision at a time")
    st.write(
        "The assistant retrieves relevant passages from the local EPLC knowledge "
        "base before it answers a policy question or drafts a document section."
    )

    qna_column, draft_column = st.columns(2)
    with qna_column:
        st.subheader("Ask a policy question")
        st.markdown(
            """
1. Ask a specific question and name the phase when possible.
2. Review the answer and its retrieved source excerpts.
3. If no relevant source is found, the assistant will say so instead of
   filling the gap with general knowledge.
"""
        )

    with draft_column:
        st.subheader("Draft a document section")
        st.markdown(
            """
1. Choose a phase and template.
2. Name one section and provide known project facts.
3. Review the draft, missing-information checklist, assumptions, and sources.
"""
        )

    st.info(
        "Avoid entering sensitive project information until an approved deployment "
        "with access control and data-handling safeguards is available."
    )


def _render_qna_page() -> None:
    st.title("Ask an EPLC policy question")
    st.caption("Answers are limited to passages retrieved from the EPLC knowledge base.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("warning"):
                st.warning(message["warning"])
            _render_citations(message.get("citations", ()))

    question = st.chat_input(
        "For example: What are the Development Phase exit criteria?"
    )
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving EPLC sources and preparing an answer…"):
                result = _api_client().answer(question)
        except Exception as exc:
            message = _friendly_error(exc)
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message}
            )
            return

        st.markdown(result.answer)
        if result.warning:
            st.warning(result.warning)
        _render_citations(result.citations)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "warning": result.warning,
                "citations": result.citations,
            }
        )


def _render_drafting_page() -> None:
    st.title("Create a grounded document section")
    st.caption(
        "The assistant drafts one section at a time and does not invent missing "
        "project facts."
    )

    phase = st.selectbox("EPLC phase", tuple(PHASE_TEMPLATES))
    templates = PHASE_TEMPLATES[phase]

    with st.form("drafting-form"):
        template = st.selectbox("Document template", templates)
        section = st.text_input(
            "Section name",
            placeholder="For example: Scope, Roles and Responsibilities, Assumptions",
        )
        project_details = st.text_area(
            "Known project details",
            height=180,
            placeholder=(
                "Describe the system, users, scope, owners, dates, dependencies, "
                "constraints, and decisions that are already known."
            ),
        )
        instructions = st.text_area(
            "Additional drafting instructions (optional)",
            height=100,
            placeholder="For example: Use a formal tone and emphasize security controls.",
        )
        submitted = st.form_submit_button(
            "Generate grounded draft",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return
    if not section.strip() or not project_details.strip():
        st.warning("Provide both a section name and known project details.")
        return

    try:
        with st.spinner("Retrieving template guidance and drafting the section…"):
            result = _api_client().draft(
                phase=phase,
                template=template,
                section=section,
                project_details=project_details,
                instructions=instructions,
            )
    except Exception as exc:
        st.error(_friendly_error(exc))
        return

    if result.warning:
        st.warning(result.warning)
        return

    st.subheader("Draft")
    st.write(result.draft)
    st.download_button(
        "Download draft as text",
        data=result.draft,
        file_name=f"{phase.lower()}-{_safe_filename(section)}.txt",
        mime="text/plain",
    )

    st.subheader("Information to confirm")
    st.write(result.missing_information)
    _render_citations(result.citations)


@st.cache_resource(show_spinner=False)
def _api_client() -> AssistantApiClient:
    return AssistantApiClient(
        base_url=os.getenv("EPLC_API_URL", "http://localhost:8000"),
    )


def _render_citations(citations: Any) -> None:
    citations = tuple(citations or ())
    if not citations:
        return

    with st.expander(f"Sources used ({len(citations)})"):
        for citation in citations:
            if isinstance(citation, Citation):
                label = f"[{citation.id}] {citation.source}"
                if citation.section:
                    label += f" — section {citation.section}"
                st.markdown(f"**{label}**")
                st.caption(citation.excerpt)


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, (ApiClientError, FileNotFoundError, RuntimeError, ValueError)):
        return str(exc)
    return (
        "The assistant could not complete the request. Check the local setup and "
        "try again."
    )


def _safe_filename(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in safe.split("-") if part) or "draft"


if __name__ == "__main__":
    main()
