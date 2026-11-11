FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PEDWRI_MODEL=/app/models/model.onnx \
    PEDWRI_OPERATING_POINTS=/app/models/operating_points.json \
    PEDWRI_OPERATING_POINT=spec_0.90

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[api]" && useradd --create-home --uid 1000 app
COPY models ./models

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "pedwri.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
