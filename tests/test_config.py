from pathlib import Path
from unittest import TestCase

from eplc_assistant.config import DEFAULT_CHROMA_ROOT, Settings


class SettingsTests(TestCase):
    def test_default_index_paths_match_repository_artifacts(self) -> None:
        settings = Settings(openai_api_key="", chroma_root=DEFAULT_CHROMA_ROOT)

        for spec in (settings.qna_index, *settings.phase_indexes.values()):
            self.assertIsInstance(spec.path, Path)
            self.assertTrue(spec.path.exists(), spec.path)
            self.assertTrue((spec.path / "chroma.sqlite3").exists(), spec.path)

    def test_api_key_validation_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            Settings(openai_api_key="").require_api_key()
