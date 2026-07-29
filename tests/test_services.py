from unittest import TestCase

from eplc_assistant.models import RetrievedChunk
from eplc_assistant.services.drafting import DraftingService
from eplc_assistant.services.qna import QnaService, REFUSAL


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []
        self.metadata_filters: list[dict | None] = []

    def retrieve(
        self,
        query: str,
        limit: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        self.metadata_filters.append(metadata_filter)
        return self.chunks[:limit]


class FakeGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
            }
        )
        return self.responses.pop(0)


def source_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        id="chunk-1",
        text="The project manager documents the phase exit criteria.",
        distance=0.1,
        index_name="eplc-framework",
        metadata={"source": "EPLC Framework", "section": "3.6"},
    )


class QnaServiceTests(TestCase):
    def test_no_context_refuses_without_calling_model(self) -> None:
        generator = FakeGenerator(["should not be used"])
        service = QnaService(
            retriever=FakeRetriever([]),
            generator=generator,
        )

        result = service.answer("What are the exit criteria?")

        self.assertEqual(REFUSAL, result.answer)
        self.assertEqual([], generator.calls)
        self.assertEqual((), result.citations)

    def test_grounded_answer_exposes_traceable_citations(self) -> None:
        generator = FakeGenerator(["The PM documents exit criteria. [S1]"])
        service = QnaService(
            retriever=FakeRetriever([source_chunk()]),
            generator=generator,
        )

        result = service.answer("What does the PM document?")

        self.assertEqual("EPLC Framework", result.citations[0].source)
        self.assertEqual("3.6", result.citations[0].section)
        self.assertIn("[S1]", result.answer)
        self.assertIn("Source: EPLC Framework", generator.calls[0]["user_prompt"])

    def test_answer_without_valid_citation_is_refused(self) -> None:
        service = QnaService(
            retriever=FakeRetriever([source_chunk()]),
            generator=FakeGenerator(["An uncited claim."]),
        )

        result = service.answer("What does the PM document?")

        self.assertEqual(REFUSAL, result.answer)
        self.assertEqual((), result.citations)
        self.assertIn("valid source citations", result.warning)


class DraftingServiceTests(TestCase):
    def test_missing_context_does_not_generate_an_ungrounded_draft(self) -> None:
        generator = FakeGenerator(["should not be used"])
        service = DraftingService(
            phase_retrievers={"design": FakeRetriever([])},
            template_sources={
                "Design": {"Test Plan": ("Test Plan Template",)}
            },
            generator=generator,
        )

        result = service.draft(
            phase="Design",
            template="Test Plan",
            section="Scope",
            project_details="A claims-processing system",
        )

        self.assertEqual("", result.draft)
        self.assertIsNotNone(result.warning)
        self.assertEqual([], generator.calls)

    def test_draft_and_completeness_check_share_grounded_context(self) -> None:
        generator = FakeGenerator(
            ["Draft text based on the project facts.", "- Testing owner"]
        )
        retriever = FakeRetriever([source_chunk()])
        service = DraftingService(
            phase_retrievers={"design": retriever},
            template_sources={
                "Design": {"Test Plan": ("Test Plan Template",)}
            },
            generator=generator,
        )

        result = service.draft(
            phase="Design",
            template="Test Plan",
            section="Scope",
            project_details="A claims-processing system",
        )

        self.assertEqual(2, len(generator.calls))
        self.assertEqual("- Testing owner", result.missing_information)
        self.assertEqual("EPLC Framework", result.citations[0].source)
        self.assertTrue(
            all(
                "The project manager documents" in call["user_prompt"]
                for call in generator.calls
            )
        )
        self.assertEqual(
            ["Test Plan Template"],
            retriever.metadata_filters[0]["$or"][0]["source"]["$in"],
        )

    def test_unsupported_template_is_rejected_before_retrieval(self) -> None:
        retriever = FakeRetriever([source_chunk()])
        service = DraftingService(
            phase_retrievers={"design": retriever},
            template_sources={
                "Design": {"Implementation Plan": ("Implementation Plan",)}
            },
            generator=FakeGenerator([]),
        )

        with self.assertRaisesRegex(ValueError, "not supported"):
            service.draft(
                phase="Design",
                template="Test Plan",
                section="Scope",
                project_details="A claims-processing system",
            )

        self.assertEqual([], retriever.queries)
