FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY ibm.py ./
COPY Data ./Data

RUN useradd --create-home appuser \
    && mkdir -p /app/var \
    && chown -R appuser:appuser /app/var

USER appuser

EXPOSE 8000

CMD ["uvicorn", "eplc_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
