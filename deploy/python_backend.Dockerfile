FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/dataana/python_backend

COPY python_backend/pyproject.toml python_backend/uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY python_backend/app ./app
COPY frontend /opt/dataana/frontend
COPY schema /opt/dataana/schema

EXPOSE 8889

ENTRYPOINT ["/opt/dataana/python_backend/.venv/bin/uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8889"]
