from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from eplc_assistant.config import Settings
from eplc_assistant.runtime import build_services


class FakeEmbedder:
    instances: list["FakeEmbedder"] = []

    def __init__(self, model_name: str, query_instruction: str = "default") -> None:
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.instances.append(self)


class FakeIndex:
    def __init__(self, spec: object) -> None:
        self.spec = spec

    def search(
        self,
        embedding: list[float],
        limit: int,
        metadata_filter: dict | None = None,
    ) -> list[object]:
        return []


class FakeGenerator:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries


class RuntimeTests(TestCase):
    @patch("eplc_assistant.runtime.OpenAITextGenerator", FakeGenerator)
    @patch("eplc_assistant.runtime.ChromaIndex", FakeIndex)
    @patch("eplc_assistant.runtime.BgeEmbeddingProvider", FakeEmbedder)
    def test_services_use_separate_qna_and_drafting_embeddings(self) -> None:
        FakeEmbedder.instances.clear()
        settings = Settings(
            openai_api_key="test-key",
            qna_embedding_model="BAAI/bge-base-en-v1.5",
            drafting_embedding_model="BAAI/bge-large-en-v1.5",
            chroma_root=Path("/indexes"),
        )

        services = build_services(settings)

        self.assertEqual(
            ("design", "development", "implementation", "requirement"),
            services.drafting.supported_phases,
        )
        self.assertEqual(6, services.qna._top_k)
        self.assertEqual(
            ["BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"],
            [embedder.model_name for embedder in FakeEmbedder.instances],
        )
        self.assertEqual("", FakeEmbedder.instances[1].query_instruction)
