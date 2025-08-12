# R3: Dockerfile for building the complete application container.
# R12: Using the Qwen3-0.6B model.
# R13: Using the correct model filename.
# R15: Forcing a generic CPU build of llama-cpp-python to prevent potential illegal instruction crashes.

# --- Stage 1: Download the Model ---
FROM python:3.10-slim as downloader

RUN pip install huggingface-hub

RUN mkdir -p /models

RUN huggingface-cli download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_S.gguf --local-dir /models --local-dir-use-symlinks False


# --- Stage 2: Build the Application ---
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# R15: FIX - Force a generic build of llama-cpp-python by disabling all common CPU-specific optimizations.
# This creates a more compatible binary that avoids SIGILL (Illegal Instruction) errors on CPUs
# that may not support these advanced instruction sets.
ENV CMAKE_ARGS="-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF"

# The pip install command will now use the CMAKE_ARGS environment variable when compiling llama-cpp-python.
RUN pip install --no-cache-dir -r requirements.txt

# Unset the variable so it doesn't affect subsequent layers if any were added.
ENV CMAKE_ARGS=""

COPY ./app /app

COPY --from=downloader /models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
