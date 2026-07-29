from unittest import TestCase

from eplc_assistant.models import RetrievedChunk
from eplc_assistant.rag import MultiIndexRetriever


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class FakeIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def search(
        self,
        embedding: list[float],
        limit: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        return self.chunks[:limit]


def chunk(chunk_id: str, distance: float, index: str = "test") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=f"text {chunk_id}",
        distance=distance,
        index_name=index,
    )


class MultiIndexRetrieverTests(TestCase):
    def test_filters_sorts_and_deduplicates_results(self) -> None:
        retriever = MultiIndexRetriever(
            embedder=FakeEmbedder(),
            indexes=[
                FakeIndex([chunk("b", 0.4), chunk("a", 0.2)]),
                FakeIndex([chunk("a", 0.2), chunk("too-far", 0.9)]),
            ],
            max_distance=0.75,
        )

        results = retriever.retrieve("question", limit=5)

        self.assertEqual(["a", "b"], [result.id for result in results])

    def test_blank_query_does_not_call_indexes(self) -> None:
        retriever = MultiIndexRetriever(
            embedder=FakeEmbedder(),
            indexes=[FakeIndex([chunk("a", 0.1)])],
            max_distance=0.75,
        )

        self.assertEqual([], retriever.retrieve("  ", limit=5))
