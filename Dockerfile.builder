# This Dockerfile is ONLY for the builder stage.
# It is used by the manual 'build-onnx-model.yml' workflow.
# Its sole purpose is to create the /onnx_model artifact.

# Start from a standard Python base image.
FROM python:3.10-slim

# Set a working directory inside the container.
WORKDIR /builder

# Install all the heavy libraries needed for the export process.
# 'optimum[exporters]' is a special package that automatically installs
# everything needed for model conversion, including 'torch', 'transformers', etc.
RUN pip install --no-cache-dir optimum[exporters]

# Copy the export script into the builder stage.
# This script contains the logic to download and convert the model.
COPY export_model.py .

# Run the export script. This will perform the download and conversion,
# and the final, clean ONNX files will be saved into the /onnx_model directory
# inside this container's filesystem.
RUN python export_model.py
