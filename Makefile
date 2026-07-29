.PHONY: install test api ui

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

api:
	uvicorn eplc_assistant.api:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ibm.py
