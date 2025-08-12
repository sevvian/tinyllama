# R3: Dockerfile for building the complete application container.
# This file defines a multi-stage process to create a clean and efficient image.

# --- Stage 1: Download the Model ---
# Use a base python image to download the model first. This helps with Docker layer caching.
FROM python:3.10-slim as downloader

# Install huggingface-hub client
RUN pip install huggingface-hub

# Create a directory for the model
RUN mkdir -p /models

# Download the specific GGUF model file. This is the core of our application's intelligence.
# This layer will be cached by Docker unless this command changes.
RUN huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir /models --local-dir-use-symlinks False


# --- Stage 2: Build the Application ---
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building llama-cpp-python
# build-essential contains the C++ compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies.
# We are not using any specific hardware acceleration, so a standard build is fine.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app

# Copy the downloaded model from the first stage into the final image
COPY --from=downloader /models /app/models

# Expose the port the application will run on
EXPOSE 8000

# The command to run the application using uvicorn web server.
# It's configured to be accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
