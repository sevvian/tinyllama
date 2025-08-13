# This is the final version of the export script.
# It performs a clean export and applies INT8 dynamic quantization to reduce file size.
import os
import shutil
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig
from onnxruntime.quantization import QuantFormat, QuantizationMode, QuantType
from transformers import AutoTokenizer, GenerationConfig

# Pin the model to the correct, latest revision for deterministic builds.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "a91318be21aeaf0879874faa161dcb40c68847e9"

# Define paths for a clean two-step process
TEMP_ONNX_PATH = "/tmp/onnx_export"
FINAL_EXPORT_PATH = "/onnx_model"

if __name__ == "__main__":
    print(f"--- Starting model preparation for '{MODEL_ID}' @ revision '{REVISION}' ---")
    
    # --- Step 1: Export the base model to a temporary ONNX path ---
    print(f"\n1. Exporting base model to temporary ONNX path: {TEMP_ONNX_PATH}")
    onnx_model = ORTModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, export=True)
    onnx_model.save_pretrained(TEMP_ONNX_PATH)
    print("Base ONNX export complete.")

    # --- Step 2: Perform EXPLICIT INT8 Dynamic Quantization ---
    print("\n2. Performing explicit INT8 dynamic quantization...")
    quantizer = ORTQuantizer.from_pretrained(TEMP_ONNX_PATH)
    
    # Create the quantization configuration explicitly.
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
        file_suffix="" # Overwrite the model.onnx file in the save_dir
    )
    print(f"Quantization complete. Quantized model saved to: {FINAL_EXPORT_PATH}")

    # --- Step 3: Save only the necessary tokenizer and config files ---
    print("\n3. Saving necessary tokenizer and config files...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    generation_config = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION)
    
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
