FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
# This will copy the model from the artifact downloaded by the workflow.
COPY onnx_model /app/model
ENV OMP_NUM_THREADS=4
ENV OMP_WAIT_POLICY=PASSIVE
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
