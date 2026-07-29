"""Composition root for production service implementations."""

from __future__ import annotations

from dataclasses import dataclass

from eplc_assistant.config import Settings
from eplc_assistant.llm import OpenAITextGenerator
from eplc_assistant.rag import BgeEmbeddingProvider, ChromaIndex, MultiIndexRetriever
from eplc_assistant.services import DraftingService, QnaService
from eplc_assistant.templates import PHASE_TEMPLATE_SOURCES


@dataclass(frozen=True)
class ApplicationServices:
    qna: QnaService
    drafting: DraftingService


def build_services(settings: Settings) -> ApplicationServices:
    """Build shared adapters once and inject them into application services."""

    settings.require_api_key()
    qna_embedder = BgeEmbeddingProvider(settings.qna_embedding_model)
    drafting_embedder = BgeEmbeddingProvider(
        settings.drafting_embedding_model,
        query_instruction="",
    )
    generator = OpenAITextGenerator(
        api_key=settings.openai_api_key,
        model=settings.chat_model,
        timeout_seconds=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )

    policy_retriever = MultiIndexRetriever(
        embedder=qna_embedder,
        indexes=[ChromaIndex(settings.qna_index)],
        max_distance=1.0 - settings.qna_min_similarity,
    )
    phase_retrievers = {
        phase: MultiIndexRetriever(
            embedder=drafting_embedder,
            indexes=[ChromaIndex(spec)],
            max_distance=settings.drafting_max_distance,
        )
        for phase, spec in settings.phase_indexes.items()
    }

    return ApplicationServices(
        qna=QnaService(
            retriever=policy_retriever,
            generator=generator,
            top_k=settings.top_k,
        ),
        drafting=DraftingService(
            phase_retrievers=phase_retrievers,
            template_sources=PHASE_TEMPLATE_SOURCES,
            generator=generator,
            top_k=settings.top_k,
        ),
    )
