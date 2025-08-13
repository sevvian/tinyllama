# This is the final version of the export script.
# It performs a clean export of the standard ONNX model WITHOUT quantization
# to ensure maximum performance and output quality on the target CPU.
import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig

# Pin the model to the correct, latest revision for deterministic builds.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "a91318be21aeaf0879874faa161dcb40c68847e9"

# Define the final export path. We no longer need a temporary path.
FINAL_EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model preparation for '{MODEL_ID}' @ revision '{REVISION}' ---")
    
    # --- Step 1: Export the base model directly to the final ONNX path ---
    print(f"\n1. Exporting base model to ONNX format at: {FINAL_EXPORT_PATH}")
    # This single command downloads, converts, and prepares the standard ONNX model.
    onnx_model = ORTModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, export=True)
    
    # --- Step 2: Save the exported model and necessary config/tokenizer files ---
    # The quantization steps have been completely removed from this script.
    print(f"\n2. Saving standard ONNX model and tokenizer to: {FINAL_EXPORT_PATH}")
    os.makedirs(FINAL_EXPORT_PATH, exist_ok=True)
    
    # Save the converted model.onnx file and its config.json
    onnx_model.save_pretrained(FINAL_EXPORT_PATH)
    
    # Separately save the tokenizer and generation config to the same directory.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    generation_config = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer.save_pretrained(FINAL_EXPORT_PATH)
    generation_config.save_pretrained(FINAL_EXPORT_PATH)
    
    print("Standard ONNX model and tokenizer saved successfully.")

    # --- Step 3: Final verification ---
    print("\n--- Final Verification ---")
    model_size = os.path.getsize(os.path.join(FINAL_EXPORT_PATH, "model.onnx")) / (1024*1024)
    print(f"Final ONNX model size: {model_size:.2f} MB")
    print(f"Final exported files in '{FINAL_EXPORT_PATH}': {os.listdir(FINAL_EXPORT_PATH)}")
    
    print("--- Model preparation complete. ---")
