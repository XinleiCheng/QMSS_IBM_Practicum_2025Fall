from pathlib import Path

from eplc_assistant.storage import InteractionRepository


def test_repository_records_and_reads_interactions(tmp_path: Path) -> None:
    repository = InteractionRepository(tmp_path / "state.sqlite3")

    interaction_id = repository.record(
        kind="qna",
        request={"question": "What are the exit criteria?"},
        response={"answer": "See section 3.5.5."},
        status="succeeded",
    )

    interaction = repository.get(interaction_id)

    assert interaction is not None
    assert interaction["request"]["question"] == "What are the exit criteria?"
    assert interaction["response"]["answer"] == "See section 3.5.5."
    assert interaction["status"] == "succeeded"


def test_recent_interactions_are_bounded(tmp_path: Path) -> None:
    repository = InteractionRepository(tmp_path / "state.sqlite3")
    for number in range(3):
        repository.record(
            kind="draft",
            request={"number": number},
            response=None,
            status="failed",
            error="test",
        )

    assert len(repository.recent(limit=2)) == 2
