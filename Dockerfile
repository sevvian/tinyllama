# R18: This is a multi-stage Dockerfile to create a thin, production-ready image.
# R20: Added Pillow to the builder stage to fix the ImportError.

# --- Stage 1: The "Builder" ---
# This stage downloads the original model and converts it to ONNX.
FROM python:3.10-slim as builder

WORKDIR /builder

# Install all libraries needed for the export process.
# R20: FIX - Added Pillow, which is required by the Idefics3ImageProcessor.
RUN pip install --no-cache-dir optimum[exporters] Pillow

# Copy the export script into the builder stage
COPY export_model.py .

# Run the export script. This will download the original model, convert it,
# and save the resulting .onnx files into the /onnx_model directory.
RUN python export_model.py


# --- Stage 2: The "Final" Application Image ---
# This stage starts fresh and only includes what's needed to run the app.
FROM python:3.10-slim

WORKDIR /app

# Install only the lightweight runtime dependencies from the corrected requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app

# Copy ONLY the exported ONNX model from the builder stage into our final image.
COPY --from=builder /onnx_model /app/model

# Expose the port the application will run on
EXPOSE 8000

# The command to run the application using uvicorn web server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
