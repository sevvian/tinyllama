# R18: This script handles the one-time conversion of the model to the ONNX format.
import os
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model export for '{MODEL_ID}' ---")

    # 1. Load the processor and save it to the export directory.
    # The runtime application will need this.
    print(f"Loading and saving processor to '{EXPORT_PATH}'...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(EXPORT_PATH)
    print("Processor saved successfully.")

    # 2. Load the original model from Hugging Face and export it to ONNX format.
    # The `export=True` flag tells optimum to perform the conversion.
    print(f"Loading and exporting model to ONNX format at '{EXPORT_PATH}'...")
    model = ORTModelForVision2Seq.from_pretrained(MODEL_ID, export=True)
    
    # 3. Save the exported ONNX model to the export directory.
    model.save_pretrained(EXPORT_PATH)
    
    print("--- Model export complete. ---")
    print(f"Exported files in '{EXPORT_PATH}': {os.listdir(EXPORT_PATH)}")
