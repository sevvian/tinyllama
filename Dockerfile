# This is the final, single-stage Dockerfile for the runtime image.
# It has NO knowledge of the builder stage.

FROM python:3.10-slim

WORKDIR /app

# Install only the lightweight runtime dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY app/ .

# Copy the pre-built ONNX model artifact from the local filesystem.
# This artifact is downloaded in the CI/CD job before this build starts.
COPY model /app/model

# Expose the port the application will run on.
EXPOSE 8000

# The command to run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
