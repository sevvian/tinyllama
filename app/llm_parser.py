# R22: This module is rewritten to use the onnxruntime library directly,
# removing the dependency on 'optimum' and 'torch' at runtime.
import json
import os
import logging
import onnxruntime as ort
from transformers import AutoTokenizer

# --- Basic Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "./model"
TOKENIZER_PATH = LOCAL_MODEL_DIR
# The exported model from optimum is typically named 'model.onnx'
ONNX_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.onnx")

class MetadataParser:
    _instance = None
    
    PROMPT_SYSTEM_INSTRUCTION = """You are an expert file name parser...""" # (Same as before, omitted for brevity)
    FEW_SHOT_EXAMPLE_1_USER = """Text: "www.Tamilblasters.qpon - Alice In Borderland (2020)..."
Output:"""
    FEW_SHOT_EXAMPLE_1_ASSISTANT = """{ "title": "Alice In Borderland", ... }"""
    FEW_SHOT_EXAMPLE_2_USER = """Text: "【高清剧集网发布 www.BPHDTV.com】外星也难民..."
Output:"""
    FEW_SHOT_EXAMPLE_2_ASSISTANT = """{ "title": "Solar Opposites", ... }"""

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MetadataParser, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'session'):
            return
            
        logger.info("--- Starting Direct ONNX Runtime Initialization ---")
        
        try:
            logger.info(f"Loading tokenizer from path: {TOKENIZER_PATH}")
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
            
            logger.info(f"Loading ONNX model and creating inference session from: {ONNX_MODEL_PATH}")
            # Use CPUExecutionProvider for our CPU-only environment
            self.session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            
            logger.info("SUCCESS: Direct ONNX Runtime session created.")
        except Exception as e:
            logger.critical("FATAL: An exception occurred during ONNX initialization.", exc_info=True)
            self.session = None
            self.tokenizer = None
        
        logger.info("--- ONNX Initialization Complete ---")

    def extract_metadata(self, title: str) -> dict:
        if not self.session or not self.tokenizer:
            # ... (error handling as before)
            return {"error": "Model/tokenizer is not available."}

        messages = [
            {"role": "system", "content": self.PROMPT_SYSTEM_INSTRUCTION},
            # ... (few-shot examples as before)
            {"role": "user", "content": f'Text: "{title}"\nOutput:'}
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="np") # Use numpy arrays for onnxruntime

            # The generate loop is now manual, as we are not using the high-level Hugging Face .generate()
            input_ids = inputs['input_ids']
            max_new_tokens = 512
            eos_token_id = self.tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                # Prepare inputs for the ONNX session
                ort_inputs = {self.session.get_inputs()[0].name: input_ids}
                # Run inference
                ort_outs = self.session.run(None, ort_inputs)
                # Get the logits for the next token
                next_token_logits = ort_outs[0][:, -1, :]
                # Greedily select the next token
                next_token = next_token_logits.argmax(axis=-1)
                
                # Append the new token
                input_ids = numpy.concatenate([input_ids, next_token[:, numpy.newaxis]], axis=-1)

                # Check for stop condition
                if next_token[0] == eos_token_id:
                    break
            
            response_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            # ... (rest of the parsing logic as before)
            
            json_start = response_text.rfind('{')
            if json_start != -1:
                json_end = response_text.rfind('}')
                if json_end > json_start:
                    response_text = response_text[json_start:json_end+1]

            parsed_json = json.loads(response_text)
            parsed_json['original_title'] = title
            return parsed_json

        except Exception as e:
            logger.error(f"An unexpected error occurred during ONNX inference: {e}", exc_info=True)
            return {"original_title": title, "error": "An unexpected error occurred."}

# Need to import numpy for the manual generation loop
import numpy
logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
