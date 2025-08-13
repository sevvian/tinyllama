# R41 (NEW): Corrected the MODEL_ID typo and removed revision pinning as per user instruction.
import os
import shutil
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from transformers import AutoTokenizer, GenerationConfig

# R41: FIX - Corrected the model name from '1M' to '135M'.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
# R41: FIX - Removed the REVISION pinning to always use the latest version.

# Define paths
TEMP_ONNX_PATH = "/tmp/onnx_export"
FINAL_EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model preparation for '{MODEL_ID}' (latest from main branch) ---")
    
    # --- Step 1: Export the base model to ONNX format ---
    print(f"\n1. Exporting base model to temporary ONNX path: {TEMP_ONNX_PATH}")
    # The 'revision' parameter has been removed from this call.
    onnx_model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    onnx_model.save_pretrained(TEMP_ONNX_PATH)
    print("Base ONNX export complete.")

    # --- Step 2: Perform EXPLICIT INT8 Dynamic Quantization ---
    print("\n2. Performing explicit INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(TEMP_ONNX_PATH)
    
    dqconfig = QuantizationConfig(
        is_static=False,
        format=QuantFormat.QOperator,
        mode=QuantizationMode.IntegerOps,
        weights_dtype=QuantType.QInt8,
        activations_dtype=QuantType.QUInt8,
        operators_to_quantize=["MatMul"]
    )

    os.makedirs(FINAL_EXPORT_PATH, exist_ok=True)
    
    quantizer.quantize(
        save_dir=FINAL_EXPORT_PATH,
        quantization_config=dqconfig,
        file_suffix="" # Overwrite the model.onnx file
    )
    print(f"Quantization complete. Quantized model saved to: {FINAL_EXPORT_PATH}")

    # --- Step 3: Save only the necessary tokenizer and config files ---
    print("\n3. Saving necessary tokenizer and config files...")
    # The 'revision' parameter has been removed from these calls.
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
