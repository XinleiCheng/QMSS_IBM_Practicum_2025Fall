import json
from unittest.mock import patch

from eplc_assistant.client import AssistantApiClient


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_frontend_client_maps_qna_response_to_domain_model() -> None:
    response = FakeResponse(
        {
            "interaction_id": "id",
            "answer": "Answer [S1]",
            "warning": None,
            "citations": [
                {
                    "id": "S1",
                    "source": "EPLC Framework",
                    "section": "3.5.5",
                    "excerpt": "Exit criteria.",
                }
            ],
        }
    )

    with patch("eplc_assistant.client.urlopen", return_value=response):
        result = AssistantApiClient().answer("What are the exit criteria?")

    assert result.answer == "Answer [S1]"
    assert result.citations[0].section == "3.5.5"
