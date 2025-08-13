# This Dockerfile is ONLY for the builder stage.
# Its sole purpose is to create the /onnx_model artifact.

FROM python:3.10-slim

WORKDIR /builder

# Install all the heavy libraries needed for the export process.
RUN pip install --no-cache-dir optimum[exporters]

# Copy the export script into the builder stage.
COPY export_model.py .

# Run the export script to create the /onnx_model directory.
RUN python export_model.py
