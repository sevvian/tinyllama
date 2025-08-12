# R3: Dockerfile for building the complete application container.
# R12: Using the Qwen3-0.6B model.
# R13: Using the correct model filename.
# R15: Forcing a generic CPU build of llama-cpp-python to prevent illegal instruction crashes.
# R16 (NEW): Forcing pip to compile from source to ensure CMAKE_ARGS are respected.

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

# R15: Set the environment variable to force a generic build of llama-cpp-python,
# specifically for CPUs like the Intel N5105 that lack AVX2 support.
ENV CMAKE_ARGS="-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF"

# R16: FIX - Use "--no-binary llama-cpp-python" to force pip to compile the library
# from source instead of using a potentially incompatible pre-compiled wheel.
# This ensures that the CMAKE_ARGS we set above are actually used during the build.
RUN pip install --no-cache-dir --no-binary llama-cpp-python -r requirements.txt

# Unset the variable so it doesn't affect subsequent layers.
ENV CMAKE_ARGS=""

COPY ./app /app

COPY --from=downloader /models /app/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
