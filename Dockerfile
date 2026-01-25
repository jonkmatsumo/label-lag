FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (changes rarely)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (cached unless deps change)
RUN uv sync --frozen --no-dev

# Copy source code last (changes frequently)
COPY src ./src
COPY config ./config
COPY scripts ./scripts

# Set up wait-for-it script
RUN cp /app/scripts/wait-for-it.sh /usr/local/bin/wait-for-it.sh && \
    chmod +x /usr/local/bin/wait-for-it.sh

# Default command
CMD ["uv", "run", "python", "-c", "print('Generator ready')"]
