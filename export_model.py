import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model export for '{MODEL_ID}' ---")

    # 1. Load the tokenizer (for text models) and save it.
    print(f"Loading and saving tokenizer to '{EXPORT_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(EXPORT_PATH)
    print("Tokenizer saved successfully.")

    # 2. Load the original model and export it to ONNX format.
    print(f"Loading and exporting model to ONNX format at '{EXPORT_PATH}'...")
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    
    # 3. Save the exported ONNX model to the export directory.
    model.save_pretrained(EXPORT_PATH)
    
    print("--- Model export complete. ---")
    print(f"Exported files in '{EXPORT_PATH}': {os.listdir(EXPORT_PATH)}")
