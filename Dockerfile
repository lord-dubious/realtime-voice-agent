# Real-Time Voice Agent Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash agent
WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy dependency files first (for caching)
COPY pyproject.toml requirements.txt requirements-dev.txt ./

# Install dependencies
RUN uv pip install --system -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Install the package in development mode
RUN uv pip install --system -e .

# Change ownership to non-root user
RUN chown -R agent:agent /app

# Switch to non-root user
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD voice-agent status || exit 1

# Default command
CMD ["voice-agent", "run"]
