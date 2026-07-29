"""Grounded EPLC policy question answering."""

from __future__ import annotations

import re
from typing import Protocol

from eplc_assistant.llm import TextGenerator
from eplc_assistant.models import AnswerResult, Citation, RetrievedChunk


REFUSAL = "Not specified in the provided EPLC sources."

SYSTEM_PROMPT = """You are an EPLC/HHS project-management assistant.
Answer only from the supplied source excerpts.
Use citations such as [S1] immediately after the claims they support.
If the excerpts do not contain the answer, reply exactly:
Not specified in the provided EPLC sources.
Do not fill gaps with general knowledge."""

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        limit: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]: ...


class QnaService:
    def __init__(
        self,
        *,
        retriever: Retriever,
        generator: TextGenerator,
        top_k: int = 6,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k

    def answer(self, question: str) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("A question is required.")

        chunks = self._retriever.retrieve(question, self._top_k)
        if not chunks:
            return AnswerResult(
                answer=REFUSAL,
                warning="No sufficiently relevant source passages were retrieved.",
            )

        context = _format_context(chunks)
        answer = self._generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"{context}\n\nQUESTION:\n{question}",
            temperature=0.0,
        )

        if answer == REFUSAL:
            return AnswerResult(
                answer=answer,
                warning="Retrieved passages were insufficient to answer this question.",
            )

        referenced_numbers = {
            int(number) for number in CITATION_PATTERN.findall(answer)
        }
        valid_numbers = set(range(1, len(chunks) + 1))
        if not referenced_numbers or not referenced_numbers <= valid_numbers:
            return AnswerResult(
                answer=REFUSAL,
                warning=(
                    "The generated answer did not contain valid source citations, "
                    "so it was not returned."
                ),
            )

        citations = tuple(
            citation
            for number, citation in enumerate(_citations(chunks), start=1)
            if number in referenced_numbers
        )
        return AnswerResult(answer=answer, citations=citations)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        section = f", section {chunk.section}" if chunk.section else ""
        blocks.append(
            f"[S{number}] Source: {chunk.source}{section}\n{chunk.text}"
        )
    return "SOURCE EXCERPTS:\n\n" + "\n\n---\n\n".join(blocks)


def _citations(chunks: list[RetrievedChunk]) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            id=f"S{number}",
            source=chunk.source,
            section=chunk.section,
            excerpt=_excerpt(chunk.text),
        )
        for number, chunk in enumerate(chunks, start=1)
    )


def _excerpt(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"
