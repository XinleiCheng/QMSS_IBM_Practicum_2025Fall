import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from eplc_assistant.api import create_app
from eplc_assistant.config import Settings
from eplc_assistant.models import AnswerResult, Citation, DraftResult
from eplc_assistant.runtime import ApplicationServices
from eplc_assistant.storage import InteractionRepository


class FakeQna:
    def answer(self, question: str) -> AnswerResult:
        return AnswerResult(
            answer="The phase requires documented exit criteria. [S1]",
            citations=(
                Citation(
                    id="S1",
                    source="EPLC Framework",
                    section="3.5.5",
                    excerpt="Exit criteria are documented.",
                ),
            ),
        )


class FakeDrafting:
    def draft(self, **request: str) -> DraftResult:
        return DraftResult(
            draft=f"Draft for {request['section']}.",
            missing_information="Project owner",
        )


def test_qna_endpoint_persists_traceable_result(tmp_path: Path) -> None:
    repository = InteractionRepository(tmp_path / "state.sqlite3")
    app = create_app(
        settings=_settings(tmp_path),
        services=ApplicationServices(qna=FakeQna(), drafting=FakeDrafting()),
        repository=repository,
    )
    async def exercise_api() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/qna",
                json={"question": "What are the Design Phase exit criteria?"},
            )
            interaction = await client.get(
                f"/api/v1/interactions/{response.json()['interaction_id']}"
            )
            return response, interaction

    response, interaction = asyncio.run(exercise_api())

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"][0]["section"] == "3.5.5"
    assert interaction.status_code == 200
    assert interaction.json()["kind"] == "qna"


def test_draft_endpoint_validates_and_returns_result(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        services=ApplicationServices(qna=FakeQna(), drafting=FakeDrafting()),
        repository=InteractionRepository(tmp_path / "state.sqlite3"),
    )
    async def exercise_api() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            invalid = await client.post(
                "/api/v1/drafts",
                json={"phase": "Design"},
            )
            response = await client.post(
                "/api/v1/drafts",
                json={
                    "phase": "Design",
                    "template": "Test Plan",
                    "section": "Scope",
                    "project_details": "A claims-processing system.",
                },
            )
            return invalid, response

    invalid, response = asyncio.run(exercise_api())

    assert invalid.status_code == 422
    assert response.status_code == 200
    assert response.json()["draft"] == "Draft for Scope."


def test_health_does_not_initialize_model_runtime(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path),
        services=SimpleNamespace(),
        repository=InteractionRepository(tmp_path / "state.sqlite3"),
    )

    async def get_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(get_health())

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="test-key",
        chroma_root=tmp_path / "indexes",
        state_database=tmp_path / "state.sqlite3",
    )
