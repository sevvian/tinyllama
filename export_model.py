# R55, R56: This script now exports the Qwen3-0.6B model with INT4 quantization.
import os
import shutil
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer, GenerationConfig

# R55: Update the model ID to Qwen3-0.6B.
MODEL_ID = "Qwen/Qwen3-0.6B"

# Define paths
TEMP_ONNX_PATH = "/tmp/onnx_export"
FINAL_EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model preparation for '{MODEL_ID}' ---")
    
    # --- Step 1: Export the base model to a temporary ONNX path ---
    print(f"\n1. Exporting base model to temporary ONNX path: {TEMP_ONNX_PATH}")
    onnx_model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    onnx_model.save_pretrained(TEMP_ONNX_PATH)
    print("Base ONNX export complete.")

    # --- Step 2: Perform INT4 Dynamic Quantization ---
    # R56: This step quantizes the model to 4-bit for better CPU performance and size.
    print("\n2. Performing INT4 dynamic block quantization...")
    quantizer = ORTQuantizer.from_pretrained(TEMP_ONNX_PATH)
    
    # The 'avx2_vnni' preset is a good choice for INT4 block quantization.
    # It targets 4-bit weights and 8-bit activations (W4A8).
    dqconfig = AutoQuantizationConfig.avx2_vnni(
        is_static=False,  # Dynamic quantization
        use_symmetric_weights=True
    )

    os.makedirs(FINAL_EXPORT_PATH, exist_ok=True)
    
    quantizer.quantize(
        save_dir=FINAL_EXPORT_PATH,
        quantization_config=dqconfig,
        file_suffix="" # Overwrite the model.onnx file
    )
    print(f"Quantization complete. Quantized model saved to: {FINAL_EXPORT_PATH}")

    # --- Step 3: Save the necessary tokenizer and config files ---
    print("\n3. Saving necessary tokenizer and config files...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    
    tokenizer.save_pretrained(FINAL_EXPORT_PATH)
    generation_config.save_pretrained(FINAL_EXPORT_PATH)
    print("Tokenizer and generation config saved.")

    # --- Step 4: Final verification and cleanup ---
    print("\n--- Final Verification ---")
    original_size = os.path.getsize(os.path.join(TEMP_ONNX_PATH, "model.onnx")) / (1024*1024)
    quantized_size = os.path.getsize(os.path.join(FINAL_EXPORT_PATH, "model.onnx")) / (1024*1024)
    print(f"Original ONNX model size: {original_size:.2f} MB")
    print(f"Quantized ONNX model size: {quantized_size:.2f} MB")
    print(f"Final exported files in '{FINAL_EXPORT_PATH}': {os.listdir(FINAL_EXPORT_PATH)}")
    
    shutil.rmtree(TEMP_ONNX_PATH)
    print("Temporary export directory cleaned up.")
    print("--- Model preparation complete. ---")
