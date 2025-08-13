# This Dockerfile is ONLY for the builder stage.
# Its sole purpose is to create the /onnx_model artifact.
FROM python:3.10-slim
WORKDIR /builder
RUN pip install --no-cache-dir optimum[exporters]
COPY export_model.py .
RUN python export_model.py
