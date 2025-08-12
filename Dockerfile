# R3: Dockerfile for building the complete application container.
# R12: Using the highly efficient Qwen3-0.6B model to fit within strict memory limits.

# --- Stage 1: Download the Model ---
FROM python:3.10-slim as downloader

RUN pip install huggingface-hub

RUN mkdir -p /models

# R12: FIX - Download the very small Qwen3-0.6B model. We use the Q4_K_S version for a good balance.
# The exact filename is "unsloth_qwen3-0.6b-q4_k_s.gguf", which we must use in the app.
RUN huggingface-cli download unsloth/Qwen3-0.6B-GGUF unsloth_qwen3-0.6b-q4_k_s.gguf --local-dir /models --local-dir-use-symlinks False


# --- Stage 2: Build the Application ---
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

COPY --from=downloader /models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
