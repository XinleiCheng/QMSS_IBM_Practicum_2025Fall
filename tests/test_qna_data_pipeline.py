import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


PIPELINE_FILE = (
    Path(__file__).parents[1]
    / "Coding"
    / "Q&A"
    / "qna_data_pipeline.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "qna_data_pipeline",
    PIPELINE_FILE,
)
pipeline = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(pipeline)


class FakeEmbeddings:
    def __init__(self, count):
        self.count = count

    def tolist(self):
        return [[0.1, 0.2, 0.3] for _ in range(self.count)]


class FakeSentenceTransformer:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def encode(self, documents, **kwargs):
        return FakeEmbeddings(len(documents))


class FakeCollection:
    def __init__(self):
        self.added = None

    def add(self, **kwargs):
        self.added = kwargs


class FakePersistentClient:
    last_instance = None

    def __init__(self, path):
        self.path = path
        self.collection = FakeCollection()
        self.collection_config = None
        FakePersistentClient.last_instance = self

    def create_collection(self, **kwargs):
        self.collection_config = kwargs
        return self.collection


class QnaDataPipelineTests(TestCase):
    def test_chunk_text_respects_limit_and_keeps_overlap(self):
        text = " ".join(
            f"Sentence {index} has five useful words."
            for index in range(20)
        )

        chunks = pipeline.chunk_text(
            text,
            max_words=30,
            overlap_words=10,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(pipeline.count_words(chunk) <= 30 for chunk in chunks)
        )
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        self.assertTrue(first_words & second_words)

    def test_stable_chunk_id_is_deterministic(self):
        first = pipeline.stable_chunk_id(
            "Framework",
            "3.5.5",
            1,
            "Exit criteria text",
        )
        second = pipeline.stable_chunk_id(
            "Framework",
            "3.5.5",
            1,
            "Exit criteria text",
        )

        self.assertEqual(first, second)

    def test_multi_paragraph_section_gets_a_traceable_overview(self):
        chunks = pipeline.section_to_chunks(
            document_title="Framework",
            source_file="source.json",
            source_type="framework",
            section_number="3.8.4",
            section_title="Deliverables",
            content=(
                "Implementation Notice: Records the deployment decision.\n\n"
                "Training Materials: Support end-user training.\n\n"
                "System Documentation: Describes the production system."
            ),
            phase="Implementation Phase",
        )

        overview = chunks[0]
        self.assertEqual(
            "section_overview",
            overview["metadata"]["chunk_kind"],
        )
        self.assertIn("Implementation Notice", overview["document"])
        self.assertIn("Training Materials", overview["document"])
        self.assertIn("System Documentation", overview["document"])

    def test_prepared_dataset_has_traceable_metadata(self):
        dataset = pipeline.prepare_dataset()

        pipeline.validate_chunks(
            dataset["chunks"],
            dataset["chunking"]["max_words"],
        )
        self.assertEqual(
            dataset["statistics"]["chunk_count"],
            len(dataset["chunks"]),
        )
        self.assertGreater(
            dataset["statistics"]["chunk_count"],
            55,
        )
        self.assertTrue(
            all(
                chunk["metadata"]["section_number"]
                for chunk in dataset["chunks"]
            )
        )
        self.assertTrue(
            all(
                chunk["metadata"]["section_title"]
                for chunk in dataset["chunks"]
            )
        )
        self.assertFalse(
            any(
                chunk["metadata"]["section_title"].endswith("Phase")
                and chunk["metadata"]["content_word_count"] < 5
                for chunk in dataset["chunks"]
            )
        )

    def test_existing_database_is_never_overwritten(self):
        with TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "existing"
            output_path.mkdir()
            with self.assertRaises(FileExistsError):
                pipeline.build_chroma_database(
                    "unused.json",
                    output_path,
                )

    def test_build_uses_normalized_embeddings_and_cosine(self):
        dataset = pipeline.prepare_dataset(max_words=180, overlap_words=30)

        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            prepared_path = temporary_path / "prepared.json"
            output_path = temporary_path / "database"
            prepared_path.write_text(
                json.dumps(dataset),
                encoding="utf-8",
            )

            fake_modules = {
                "chromadb": SimpleNamespace(
                    PersistentClient=FakePersistentClient
                ),
                "sentence_transformers": SimpleNamespace(
                    SentenceTransformer=FakeSentenceTransformer
                ),
            }
            with patch.dict(sys.modules, fake_modules):
                manifest = pipeline.build_chroma_database(
                    prepared_path,
                    output_path,
                )

            client = FakePersistentClient.last_instance
            self.assertEqual(
                "cosine",
                client.collection_config["metadata"]["hnsw:space"],
            )
            self.assertEqual(
                len(dataset["chunks"]),
                len(client.collection.added["ids"]),
            )
            self.assertEqual(3, manifest["embedding_dimension"])
            self.assertTrue((output_path / "manifest.json").exists())
            self.assertTrue((output_path / "embeddings.json").exists())
