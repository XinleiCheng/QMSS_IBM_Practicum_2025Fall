"""Data contracts shared by retrieval, services, and user interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    distance: float
    index_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        for key in ("document_title", "source", "title", "document"):
            value = self.metadata.get(key)
            if value:
                return str(value)
        return self.index_name

    @property
    def section(self) -> str | None:
        value = self.metadata.get("section_number") or self.metadata.get("section")
        return str(value) if value else None

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


@dataclass(frozen=True)
class Citation:
    id: str
    source: str
    excerpt: str
    section: str | None = None


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: tuple[Citation, ...] = ()
    warning: str | None = None


@dataclass(frozen=True)
class DraftResult:
    draft: str
    missing_information: str
    citations: tuple[Citation, ...] = ()
    warning: str | None = None
