# R21: This script handles the text-only SmolLM2 model.
# R24: Added a two-step process to log original model file sizes before ONNX conversion.
import os
import shutil
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
ORIGINAL_MODEL_PATH = "/original_model"
EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model preparation for '{MODEL_ID}' ---")

    # --- Step 1: Download and save the original PyTorch model ---
    print(f"Downloading original PyTorch model and tokenizer to '{ORIGINAL_MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    
    tokenizer.save_pretrained(ORIGINAL_MODEL_PATH)
    model.save_pretrained(ORIGINAL_MODEL_PATH)
    print("Original model and tokenizer downloaded and saved successfully.")

    # --- Step 2: Log the sizes of the original files ---
    # R24: This section fulfills the requirement to log file sizes.
    print("\n--- Analyzing original model file sizes ---")
    total_size_bytes = 0
    try:
        file_list = os.listdir(ORIGINAL_MODEL_PATH)
        for filename in file_list:
            filepath = os.path.join(ORIGINAL_MODEL_PATH, filename)
            if os.path.isfile(filepath):
                size_bytes = os.path.getsize(filepath)
                size_mb = size_bytes / (1024 * 1024)
                total_size_bytes += size_bytes
                print(f"  - File: {filename:<30} Size: {size_mb:.2f} MB")
        
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total original model size: {total_size_mb:.2f} MB")
    except Exception as e:
        print(f"Could not analyze file sizes: {e}")
    print("--- End of size analysis ---\n")

    # --- Step 3: Export the now-local model to ONNX format ---
    print(f"Exporting local model from '{ORIGINAL_MODEL_PATH}' to ONNX format at '{EXPORT_PATH}'...")
    # The tokenizer is already compatible, so we just copy it.
    shutil.copytree(ORIGINAL_MODEL_PATH, EXPORT_PATH, dirs_exist_ok=True)
    
    # Export the model using the local path as the source.
    onnx_model = ORTModelForCausalLM.from_pretrained(ORIGINAL_MODEL_PATH, export=True)
    
    # Save the exported ONNX model files, overwriting the pytorch ones in the export path.
    onnx_model.save_pretrained(EXPORT_PATH)
    
    print("--- Model export complete. ---")
    final_files = os.listdir(EXPORT_PATH)
    print(f"Final exported files in '{EXPORT_PATH}': {final_files}")
