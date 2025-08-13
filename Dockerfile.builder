# This Dockerfile is ONLY for the builder stage.
FROM python:3.10-slim
WORKDIR /builder
COPY requirements-builder.txt .
RUN pip install --no-cache-dir -r requirements-builder.txt
COPY export_model.py .
RUN python export_model.py
