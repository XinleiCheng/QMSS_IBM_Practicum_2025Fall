"""Small HTTP client used by the Streamlit frontend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from eplc_assistant.models import AnswerResult, Citation, DraftResult


class ApiClientError(RuntimeError):
    """A user-safe backend communication error."""


@dataclass(frozen=True)
class AssistantApiClient:
    base_url: str = "http://localhost:8000"
    timeout_seconds: float = 120.0

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def answer(self, question: str) -> AnswerResult:
        payload = self._request(
            "POST",
            "/api/v1/qna",
            {"question": question},
        )
        return AnswerResult(
            answer=payload["answer"],
            warning=payload.get("warning"),
            citations=_citations(payload.get("citations", [])),
        )

    def draft(
        self,
        *,
        phase: str,
        template: str,
        section: str,
        project_details: str,
        instructions: str = "",
    ) -> DraftResult:
        payload = self._request(
            "POST",
            "/api/v1/drafts",
            {
                "phase": phase,
                "template": template,
                "section": section,
                "project_details": project_details,
                "instructions": instructions,
            },
        )
        return DraftResult(
            draft=payload["draft"],
            missing_information=payload["missing_information"],
            warning=payload.get("warning"),
            citations=_citations(payload.get("citations", [])),
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = (
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _http_error_detail(exc)
            raise ApiClientError(detail) from exc
        except URLError as exc:
            raise ApiClientError(
                "The EPLC API is unavailable. Start the backend and try again."
            ) from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise ApiClientError(
                "The EPLC API returned an invalid or timed-out response."
            ) from exc


def _citations(items: list[dict[str, Any]]) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            id=item["id"],
            source=item["source"],
            excerpt=item["excerpt"],
            section=item.get("section"),
        )
        for item in items
    )


def _http_error_detail(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read().decode("utf-8"))
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    return f"The EPLC API returned HTTP {exc.code}."
