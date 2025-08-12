# R3: Dockerfile for building the complete application container.
# R12: Using the highly efficient Qwen3-0.6B model.
# R13: Corrected the model filename to prevent 404 errors during download.

# --- Stage 1: Download the Model ---
FROM python:3.10-slim as downloader

RUN pip install huggingface-hub

RUN mkdir -p /models

# R13: FIX - Using the correct filename "Qwen3-0.6B-Q4_K_S.gguf" as found on the Hugging Face repo.
RUN huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_S.gguf --local-dir /models --local-dir-use-symlinks False


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
