"""FastAPI transport for the existing EPLC application services."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from eplc_assistant.config import Settings
from eplc_assistant.runtime import ApplicationServices, build_services
from eplc_assistant.storage import InteractionRepository


logger = logging.getLogger("eplc_assistant.api")
API_VERSION = "1.0.0"


class QnaRequest(BaseModel):
    question: str = Field(min_length=1)


class DraftRequest(BaseModel):
    phase: str = Field(min_length=1, max_length=50)
    template: str = Field(min_length=1, max_length=200)
    section: str = Field(min_length=1, max_length=200)
    project_details: str = Field(min_length=1, max_length=12_000)
    instructions: str = Field(default="", max_length=4_000)


class CitationResponse(BaseModel):
    id: str
    source: str
    excerpt: str
    section: str | None = None


class QnaResponse(BaseModel):
    interaction_id: str
    answer: str
    citations: list[CitationResponse]
    warning: str | None = None


class DraftResponse(BaseModel):
    interaction_id: str
    draft: str
    missing_information: str
    citations: list[CitationResponse]
    warning: str | None = None


def create_app(
    *,
    settings: Settings | None = None,
    services: ApplicationServices | None = None,
    repository: InteractionRepository | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="EPLC AI Assistant API",
        version=API_VERSION,
        description="Grounded EPLC guidance and document-drafting API.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @lru_cache(maxsize=1)
    def service_provider() -> ApplicationServices:
        return services or build_services(settings)

    @lru_cache(maxsize=1)
    def repository_provider() -> InteractionRepository:
        return repository or InteractionRepository(settings.state_database)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": API_VERSION}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        index_paths = [
            settings.qna_index.path,
            *(spec.path for spec in settings.phase_indexes.values()),
        ]
        missing_indexes = [
            str(path)
            for path in index_paths
            if not (path / "chroma.sqlite3").is_file()
        ]
        state_database_available = True
        try:
            repository_provider()
        except (OSError, sqlite3.Error):
            state_database_available = False
        checks = {
            "api_key_configured": bool(settings.openai_api_key),
            "indexes_available": not missing_indexes,
            "state_database_available": state_database_available,
        }
        if not all(checks.values()):
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "checks": checks},
            )
        return {"status": "ready", "checks": checks}

    @app.post("/api/v1/qna", response_model=QnaResponse)
    def answer_question(request: QnaRequest) -> QnaResponse:
        payload = _model_dict(request)
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="A question is required.")
        if len(question) > settings.max_question_length:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Question exceeds the {settings.max_question_length}-character "
                    "limit."
                ),
            )
        try:
            service = _get_services(service_provider)
            result = service.qna.answer(question)
            response_payload = {
                "answer": result.answer,
                "citations": [asdict(citation) for citation in result.citations],
                "warning": result.warning,
            }
            interaction_id = repository_provider().record(
                kind="qna",
                request=payload,
                response=response_payload,
                status="refused" if result.warning else "succeeded",
            )
            return QnaResponse(
                interaction_id=interaction_id,
                **response_payload,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            _record_failure(repository_provider, "qna", payload, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            _record_failure(repository_provider, "qna", payload, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Q&A request failed")
            _record_failure(repository_provider, "qna", payload, exc)
            raise HTTPException(
                status_code=500,
                detail="The assistant could not complete the request.",
            ) from exc

    @app.post("/api/v1/drafts", response_model=DraftResponse)
    def create_draft(request: DraftRequest) -> DraftResponse:
        payload = _model_dict(request)
        try:
            service = _get_services(service_provider)
            result = service.drafting.draft(**payload)
            response_payload = {
                "draft": result.draft,
                "missing_information": result.missing_information,
                "citations": [asdict(citation) for citation in result.citations],
                "warning": result.warning,
            }
            interaction_id = repository_provider().record(
                kind="draft",
                request=payload,
                response=response_payload,
                status="refused" if result.warning else "succeeded",
            )
            return DraftResponse(
                interaction_id=interaction_id,
                **response_payload,
            )
        except ValueError as exc:
            _record_failure(repository_provider, "draft", payload, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            _record_failure(repository_provider, "draft", payload, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Draft request failed")
            _record_failure(repository_provider, "draft", payload, exc)
            raise HTTPException(
                status_code=500,
                detail="The assistant could not complete the request.",
            ) from exc

    @app.get("/api/v1/interactions")
    def list_interactions(
        limit: int = Query(default=20, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        return repository_provider().recent(limit)

    @app.get("/api/v1/interactions/{interaction_id}")
    def get_interaction(interaction_id: str) -> dict[str, Any]:
        interaction = repository_provider().get(interaction_id)
        if interaction is None:
            raise HTTPException(status_code=404, detail="Interaction not found.")
        return interaction

    return app


def _record_failure(
    repository_provider: Callable[[], InteractionRepository],
    kind: str,
    request: dict[str, Any],
    exc: Exception,
) -> None:
    try:
        repository_provider().record(
            kind=kind,
            request=request,
            response=None,
            status="failed",
            error=str(exc),
        )
    except Exception:
        logger.exception("Could not persist failed interaction")


def _get_services(
    service_provider: Callable[[], ApplicationServices],
) -> ApplicationServices:
    try:
        return service_provider()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _model_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


app = create_app()
