# R27, R28, R29: This script now performs a clean export, adds INT8 quantization,
# and pins the model revision for reproducibility.
import os
import shutil
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig, QuantizationType
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# R29: Pin the model and revision for deterministic builds.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "e22a5366c348009257db3cf9d0526315530424f3" # Pinned to a specific commit

# Define paths
TEMP_ONNX_PATH = "/tmp/onnx_export"
FINAL_EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model preparation for '{MODEL_ID}' @ revision '{REVISION}' ---")
    
    # --- Step 1: Export the base model to ONNX format ---
    print(f"\n1. Exporting base model to temporary ONNX path: {TEMP_ONNX_PATH}")
    onnx_model = ORTModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, export=True)
    onnx_model.save_pretrained(TEMP_ONNX_PATH)
    print("Base ONNX export complete.")

    # --- Step 2: Perform INT8 Dynamic Quantization ---
    # R28: This step quantizes the model for better CPU performance.
    print("\n2. Performing INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(TEMP_ONNX_PATH)
    dqconfig = QuantizationConfig(
        quantization_type=QuantizationType.DYNAMIC,
        per_channel=False,
        operators_to_quantize=["MatMul"] # Quantize the most impactful operators.
    )
    
    # The quantized model will be saved directly to our clean final path.
    quantized_model_path = os.path.join(FINAL_EXPORT_PATH, "model.onnx")
    os.makedirs(FINAL_EXPORT_PATH, exist_ok=True)
    
    quantizer.quantize(
        save_dir=FINAL_EXPORT_PATH,
        quantization_config=dqconfig,
        file_suffix="" # Overwrite the model.onnx file in the save_dir
    )
    print(f"Quantization complete. Quantized model saved to: {FINAL_EXPORT_PATH}")

    # --- Step 3: Save only the necessary tokenizer and config files ---
    # R27: This ensures a clean export directory without any PyTorch files.
    print("\n3. Saving necessary tokenizer and config files...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    generation_config = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION)
    
    # Save these files to the same final directory.
    tokenizer.save_pretrained(FINAL_EXPORT_PATH)
    generation_config.save_pretrained(FINAL_EXPORT_PATH)
    # The main model config was already saved by the quantizer.
    print("Tokenizer and generation config saved.")

    # --- Step 4: Final verification and cleanup ---
    print("\n--- Final Verification ---")
    original_size = os.path.getsize(os.path.join(TEMP_ONNX_PATH, "model.onnx")) / (1024*1024)
    quantized_size = os.path.getsize(os.path.join(FINAL_EXPORT_PATH, "model.onnx")) / (1024*1024)
    print(f"Original ONNX model size: {original_size:.2f} MB")
    print(f"Quantized ONNX model size: {quantized_size:.2f} MB")
    print(f"Final exported files in '{FINAL_EXPORT_PATH}': {os.listdir(FINAL_EXPORT_PATH)}")
    
    # Clean up the temporary directory
    shutil.rmtree(TEMP_ONNX_PATH)
    print("Temporary export directory cleaned up.")
    print("--- Model preparation complete. ---")
