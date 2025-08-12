# R17: Overhauled Dockerfile for the ONNX Runtime stack.
# This version is simpler as it does not require C++ build tools or manual model downloads.

FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install the new set of Python dependencies.
# No C++ compilation is needed, making this step faster and more reliable.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app

# The transformers library will automatically download and cache the model on first run.
# We expose the cache directory as a volume so the model is not re-downloaded on every container restart.
# This is an optional but recommended step for performance.
VOLUME /root/.cache/huggingface

# Expose the port the application will run on
EXPOSE 8000

# The command to run the application using uvicorn web server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
