FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/dataana/backend

COPY backend/pyproject.toml backend/uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY backend/app ./app
COPY frontend /opt/dataana/frontend
COPY schema /opt/dataana/schema

EXPOSE 8889

ENTRYPOINT ["/opt/dataana/backend/.venv/bin/uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8889"]
