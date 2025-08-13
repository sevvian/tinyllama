# This is the final, single-stage Dockerfile for the runtime image.
# It assumes the ONNX model files already exist in the repository.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# This command copies the model from the checked-out repository into the image.
COPY onnx_model /app/model

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
