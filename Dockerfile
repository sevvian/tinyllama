# This is the final, single-stage Dockerfile for the runtime image.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY onnx_model /app/model

# R31: Set environment variables for ONNX Runtime performance tuning on the N5105 CPU.
ENV OMP_NUM_THREADS=4
ENV OMP_WAIT_POLICY=PASSIVE

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
