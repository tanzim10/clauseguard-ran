# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir -e .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "clauseguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
