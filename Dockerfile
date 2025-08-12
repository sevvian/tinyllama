# R18: This is a multi-stage Dockerfile to create a thin, production-ready image.

# --- Stage 1: The "Builder" ---
# This stage downloads the original model and converts it to ONNX.
FROM python:3.10-slim as builder

WORKDIR /builder

# Install all libraries needed for the export process
# We need torch for the original model and optimum for the conversion tool.
RUN pip install --no-cache-dir transformers torch optimum[exporters]

# Copy the export script into the builder stage
COPY export_model.py .

# Run the export script. This will download the original model, convert it,
# and save the resulting .onnx files into the /onnx_model directory.
RUN python export_model.py


# --- Stage 2: The "Final" Application Image ---
# This stage starts fresh and only includes what's needed to run the app.
FROM python:3.10-slim

WORKDIR /app

# Install only the lightweight runtime dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app

# Copy ONLY the exported ONNX model from the builder stage into our final image.
# This is the key step that keeps the final image small.
COPY --from=builder /onnx_model /app/model

# Expose the port the application will run on
EXPOSE 8000

# The command to run the application using uvicorn web server.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
