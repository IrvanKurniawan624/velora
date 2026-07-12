# Stage 1: Build virtual environment and compile native extensions
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install compilation dependencies for compiling native packages like llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev


# Stage 2: Final lightweight runner image
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install runtime utilities + OpenMP (required by llama-cpp-python for local model inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Setup grading volume mount points and model directory
RUN mkdir -p /input /output /app/models

# Download local Qwen 2.5 3B Q4_K_M GGUF model during build
RUN curl -L -o /app/models/model.gguf \
    https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

# Copy codebase
COPY . .

# Set execution path to use python from virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Run main agent module
CMD ["python", "-m", "app.main"]
