# This is the final, single-stage Dockerfile for the runtime image.
FROM python:3.10-slim

WORKDIR /app

# This uses the minimal runtime requirements.txt file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .
COPY onnx_model /app/model

ENV OMP_NUM_THREADS=4
ENV OMP_WAIT_POLICY=PASSIVE

EXPOSE 8000

# R49: FIX - Add the '--timeout-keep-alive 300' flag to the Uvicorn command.
# This increases the server's timeout to 5 minutes to prevent the frontend
# from disconnecting during slow, CPU-intensive model inference.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
