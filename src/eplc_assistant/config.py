"""Central runtime configuration and local persistence locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_ROOT = REPOSITORY_ROOT / "Data" / "Vector DataBase"
DEFAULT_STATE_DATABASE = REPOSITORY_ROOT / "var" / "eplc_assistant.sqlite3"


@dataclass(frozen=True)
class IndexSpec:
    """A named Chroma index used by a retrieval workflow."""

    name: str
    path: Path


@dataclass(frozen=True)
class Settings:
    """Application settings loaded once at the composition boundary."""

    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    qna_embedding_model: str = "BAAI/bge-base-en-v1.5"
    drafting_embedding_model: str = "BAAI/bge-large-en-v1.5"
    top_k: int = 6
    qna_min_similarity: float = 0.61
    drafting_max_distance: float = 0.75
    max_question_length: int = 1000
    openai_timeout_seconds: float = 60.0
    openai_max_retries: int = 2
    chroma_root: Path = DEFAULT_CHROMA_ROOT
    state_database: Path = DEFAULT_STATE_DATABASE
    cors_origins: tuple[str, ...] = ("http://localhost:8501",)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load `.env` when available, then construct validated settings."""

        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None

        if load_dotenv is not None:
            load_dotenv(REPOSITORY_ROOT / ".env")

        chroma_value = os.getenv("CHROMA_ROOT", "").strip()
        chroma_root = Path(chroma_value).expanduser() if chroma_value else DEFAULT_CHROMA_ROOT
        if not chroma_root.is_absolute():
            chroma_root = REPOSITORY_ROOT / chroma_root

        state_value = os.getenv("STATE_DATABASE", "").strip()
        state_database = (
            Path(state_value).expanduser()
            if state_value
            else DEFAULT_STATE_DATABASE
        )
        if not state_database.is_absolute():
            state_database = REPOSITORY_ROOT / state_database

        cors_origins = tuple(
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:8501",
            ).split(",")
            if origin.strip()
        )

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini").strip(),
            qna_embedding_model=os.getenv(
                "QNA_EMBEDDING_MODEL",
                os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            ).strip(),
            drafting_embedding_model=os.getenv(
                "DRAFTING_EMBEDDING_MODEL",
                "BAAI/bge-large-en-v1.5",
            ).strip(),
            top_k=int(os.getenv("TOP_K", "6")),
            qna_min_similarity=float(os.getenv("MIN_SIMILARITY", "0.61")),
            drafting_max_distance=float(
                os.getenv("DRAFTING_MAX_DISTANCE", "0.75")
            ),
            max_question_length=int(os.getenv("MAX_QUESTION_LENGTH", "1000")),
            openai_timeout_seconds=float(
                os.getenv("OPENAI_TIMEOUT_SECONDS", "60")
            ),
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
            chroma_root=chroma_root.resolve(),
            state_database=state_database.resolve(),
            cors_origins=cors_origins,
        )

    def require_api_key(self) -> None:
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and add a key."
            )

    @property
    def qna_index(self) -> IndexSpec:
        return IndexSpec(
            "qna-knowledge-v4",
            self.chroma_root / "qna_v4_bge_base_en_v1_5",
        )

    @property
    def phase_indexes(self) -> dict[str, IndexSpec]:
        return {
            "requirement": IndexSpec(
                "requirement-templates",
                self.chroma_root / "chroma_db_Requirement Phase",
            ),
            "design": IndexSpec(
                "design-templates",
                self.chroma_root / "chroma_db_Design Phase",
            ),
            "development": IndexSpec(
                "development-templates",
                self.chroma_root / "chroma_db_development_phase",
            ),
            "implementation": IndexSpec(
                "implementation-templates",
                self.chroma_root / "chroma_db_Implementation_Phase",
            ),
        }
