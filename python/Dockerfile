FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

ARG UV_VERSION=0.9.22
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev

COPY python/src ./src
COPY python/config ./config

ENV PYTHONPATH=/app/src

EXPOSE 50052

CMD ["uv", "run", "python", "-m", "inference.server"]
