import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase


QNA_FILE = (
    Path(__file__).parents[1]
    / "Coding"
    / "Q&A"
    / "qna_finalv2.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qna_finalv2",
    QNA_FILE,
)
qna = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(qna)


class FakeEmbedding:
    def tolist(self):
        return [[0.1, 0.2]]


class FakeEmbeddingModel:
    def encode(self, texts, normalize_embeddings):
        return FakeEmbedding()


class FakeCollection:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error

    def query(self, **kwargs):
        if self.error:
            raise self.error
        return {
            "ids": [[result["id"] for result in self.results]],
            "documents": [
                [result["document"] for result in self.results]
            ],
            "distances": [
                [result["distance"] for result in self.results]
            ],
            "metadatas": [
                [result.get("metadata", {}) for result in self.results]
            ],
        }


class FakeResponses:
    def __init__(self, output_text=None, error=None):
        self.output_text = output_text
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.output_text)


class FakeOpenAIClient:
    def __init__(self, output_text=None, error=None):
        self.responses = FakeResponses(output_text, error)


def source_result(distance=0.2):
    return {
        "id": "design-exit-criteria",
        "document": (
            "The project manager documents the Design Phase "
            "exit criteria."
        ),
        "distance": distance,
        "metadata": {
            "source": "EPLC Framework",
            "section_number": "3.6",
            "title": "Exit Criteria",
        },
    }


def collections(eplc_results=None, hhs_results=None):
    return {
        "EPLC": FakeCollection(eplc_results),
        "HHS": FakeCollection(hhs_results),
    }


class QnaFinalV2Tests(TestCase):
    def test_grounded_answer_has_traceable_source(self):
        client = FakeOpenAIClient(
            "The project manager documents the exit criteria "
            "[Source 1]."
        )

        result = qna.answer_question(
            "Who documents the exit criteria?",
            FakeEmbeddingModel(),
            collections([source_result()]),
            client,
        )

        self.assertEqual("answered", result["status"])
        self.assertEqual(1, result["retrieval_count"])
        self.assertEqual(1, result["sources"][0]["number"])
        self.assertIn(
            "EPLC Framework | 3.6 | Exit Criteria",
            result["sources"][0]["citation"],
        )
        request = client.responses.calls[0]
        self.assertEqual(qna.SYSTEM_PROMPT, request["instructions"])
        self.assertIn("[Source 1]", request["input"])

    def test_no_relevant_context_refuses_without_openai_call(self):
        client = FakeOpenAIClient("This response must not be used.")

        result = qna.answer_question(
            "What is the company vacation policy?",
            FakeEmbeddingModel(),
            collections([source_result(distance=0.9)]),
            client,
        )

        self.assertEqual("not_found", result["status"])
        self.assertEqual(qna.REFUSAL_ANSWER, result["answer"])
        self.assertEqual([], result["sources"])
        self.assertEqual([], client.responses.calls)

    def test_model_refusal_does_not_use_general_knowledge(self):
        client = FakeOpenAIClient(qna.REFUSAL_ANSWER)

        result = qna.answer_question(
            "Question not answered by the retrieved passage",
            FakeEmbeddingModel(),
            collections([source_result()]),
            client,
        )

        self.assertEqual("not_found", result["status"])
        self.assertEqual([], result["sources"])
        self.assertEqual(1, len(client.responses.calls))

    def test_openai_error_is_returned_as_system_error(self):
        client = FakeOpenAIClient(error=RuntimeError("secret detail"))

        result = qna.answer_question(
            "Who documents the exit criteria?",
            FakeEmbeddingModel(),
            collections([source_result()]),
            client,
        )

        self.assertEqual("error", result["status"])
        self.assertNotIn("secret detail", result["answer"])
        self.assertEqual([], result["sources"])

    def test_answer_without_source_reference_is_rejected(self):
        client = FakeOpenAIClient(
            "The project manager documents the exit criteria."
        )

        result = qna.answer_question(
            "Who documents the exit criteria?",
            FakeEmbeddingModel(),
            collections([source_result()]),
            client,
        )

        self.assertEqual("error", result["status"])
        self.assertEqual([], result["sources"])

    def test_unknown_source_reference_is_rejected(self):
        client = FakeOpenAIClient(
            "The project manager documents the exit criteria [Source 2]."
        )

        result = qna.answer_question(
            "Who documents the exit criteria?",
            FakeEmbeddingModel(),
            collections([source_result()]),
            client,
        )

        self.assertEqual("error", result["status"])
        self.assertEqual([], result["sources"])

    def test_retrieval_error_is_returned_as_system_error(self):
        broken_collections = {
            "EPLC": FakeCollection(error=RuntimeError("database failure")),
            "HHS": FakeCollection(),
        }

        result = qna.answer_question(
            "Who documents the exit criteria?",
            FakeEmbeddingModel(),
            broken_collections,
            FakeOpenAIClient(),
        )

        self.assertEqual("error", result["status"])
        self.assertIn("retrieval service", result["answer"])

    def test_blank_question_is_invalid(self):
        result = qna.answer_question(
            "   ",
            FakeEmbeddingModel(),
            collections(),
            FakeOpenAIClient(),
        )

        self.assertEqual("invalid_request", result["status"])
        self.assertEqual("Please enter a question.", result["answer"])

    def test_question_length_is_limited(self):
        result = qna.answer_question(
            "x" * (qna.MAX_QUESTION_LENGTH + 1),
            FakeEmbeddingModel(),
            collections(),
            FakeOpenAIClient(),
        )

        self.assertEqual("invalid_request", result["status"])
        self.assertIn("character limit", result["answer"])
