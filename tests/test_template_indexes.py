import sqlite3

from eplc_assistant.config import Settings
from eplc_assistant.templates import PHASE_TEMPLATE_SOURCES


def test_every_frontend_template_exists_in_its_vector_index() -> None:
    settings = Settings(openai_api_key="")

    for phase, templates in PHASE_TEMPLATE_SOURCES.items():
        spec = settings.phase_indexes[phase.lower()]
        database_uri = f"file:{(spec.path / 'chroma.sqlite3').resolve()}?mode=ro"
        connection = sqlite3.connect(database_uri, uri=True)
        for template, source_names in templates.items():
            placeholders = ", ".join("?" for _ in source_names)
            result = connection.execute(
                f"""
                SELECT 1
                FROM embedding_metadata
                WHERE key IN ('source', 'document')
                  AND string_value IN ({placeholders})
                LIMIT 1
                """,
                source_names,
            ).fetchone()
            assert result, (
                f"{phase}/{template} has no matching chunks in {spec.path}"
            )
        connection.close()
