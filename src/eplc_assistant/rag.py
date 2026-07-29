"""Embedding and vector-retrieval components shared by all workflows."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

from eplc_assistant.config import IndexSpec
from eplc_assistant.models import RetrievedChunk

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class EmbeddingProvider(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class SearchableIndex(Protocol):
    def search(
        self,
        embedding: Sequence[float],
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


class BgeEmbeddingProvider:
    """Query encoder matching the BGE model used to build the checked-in indexes."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        query_instruction: str = "Represent this sentence for searching relevant passages:",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self._query_instruction = query_instruction.strip()

    def embed_query(self, text: str) -> list[float]:
        instructed_text = (
            f"{self._query_instruction} {text}"
            if self._query_instruction
            else text
        )
        vectors = self._model.encode(
            [instructed_text],
            normalize_embeddings=True,
        )
        return vectors[0].tolist()


class ChromaIndex:
    """Read-only adapter around one persisted Chroma collection."""

    def __init__(self, spec: IndexSpec) -> None:
        if not spec.path.exists():
            raise FileNotFoundError(
                f"Vector index '{spec.name}' was not found at {spec.path}"
            )

        from chromadb import PersistentClient

        client = PersistentClient(path=str(spec.path))
        collections = client.list_collections()
        if len(collections) != 1:
            names = [collection.name for collection in collections]
            raise RuntimeError(
                f"Vector index '{spec.name}' must contain exactly one collection; "
                f"found {names or 'none'}."
            )

        self._collection = collections[0]
        self._spec = spec

    def search(
        self,
        embedding: Sequence[float],
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        query: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            query["where"] = metadata_filter
        result = self._collection.query(
            **query,
        )

        ids = _first_result(result.get("ids"))
        documents = _first_result(result.get("documents"))
        metadatas = _first_result(result.get("metadatas"))
        distances = _first_result(result.get("distances"))

        chunks: list[RetrievedChunk] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if not document:
                continue
            chunks.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=str(document),
                    distance=float(distance),
                    index_name=self._spec.name,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks


class MultiIndexRetriever:
    """Search several indexes and return one deduplicated relevance ranking."""

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        indexes: Sequence[SearchableIndex],
        max_distance: float,
    ) -> None:
        self._embedder = embedder
        self._indexes = tuple(indexes)
        self._max_distance = max_distance

    def retrieve(
        self,
        query: str,
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        embedding = self._embedder.embed_query(query)
        candidates = [
            chunk
            for index in self._indexes
            for chunk in index.search(embedding, limit, metadata_filter)
            if chunk.distance <= self._max_distance
        ]
        candidates.sort(key=lambda chunk: chunk.distance)

        unique: list[RetrievedChunk] = []
        seen: set[tuple[str, str]] = set()
        for chunk in candidates:
            key = (chunk.index_name, chunk.id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)
            if len(unique) == limit:
                break
        return unique


def _first_result(value: Any) -> list[Any]:
    if not value:
        return []
    return list(value[0])
