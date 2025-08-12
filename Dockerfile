# R18: Multi-stage Dockerfile.
# R21: Using the text-only SmolLM2 model.

# --- Stage 1: The "Builder" ---
# This stage downloads the original model and converts it to ONNX.
FROM python:3.10-slim as builder

WORKDIR /builder

# Install libraries for the export process.
RUN pip install --no-cache-dir optimum[exporters]

# Copy the export script into the builder stage
COPY export_model.py .

# Run the export script.
RUN python export_model.py


# --- Stage 2: The "Final" Application Image ---
# This stage starts fresh and only includes what's needed to run the app.
FROM python:3.10-slim

WORKDIR /app

# Install only the lightweight runtime dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# R26: FIX - Copy the *contents* of the app directory to the WORKDIR.
# This creates a clean /app/main.py, /app/static/ structure.
COPY app/ .

# Copy ONLY the exported ONNX model from the builder stage into our final image.
COPY --from=builder /onnx_model /app/model

# Expose the port the application will run on
EXPOSE 8000

# The command to run the application using uvicorn web server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
