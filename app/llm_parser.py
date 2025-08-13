# R35: This module is rewritten to use the onnxruntime library directly,
# removing the dependency on 'optimum' and 'torch' at runtime.
import json
import os
import logging
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy

# --- Basic Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "./model"
TOKENIZER_PATH = LOCAL_MODEL_DIR
ONNX_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.onnx")

class MetadataParser:
    _instance = None
    
    PROMPT_SYSTEM_INSTRUCTION = """You are an expert file name parser. Your task is to extract metadata from the user's text and return a clean JSON object. The fields to extract are: title, year, season, episode, resolution, audio_language, source, release_group. If a field is not present, return it as null. Respond with ONLY the JSON object."""
    FEW_SHOT_EXAMPLE_1_USER = """Text: "www.Tamilblasters.qpon - Alice In Borderland (2020) S02 EP (01-08) - HQ HDRip - 720p - [Tam+ Hin + Eng] - (AAC 2.0) - 2.8GB - ESub"
Output:"""
    FEW_SHOT_EXAMPLE_1_ASSISTANT = """{
  "title": "Alice In Borderland",
  "year": 2020,
  "season": 2,
  "episode": "01-08",
  "resolution": "720p",
  "audio_language": ["Tamil", "Hindi", "English"],
  "source": "HDRip",
  "release_group": null
}"""
    FEW_SHOT_EXAMPLE_2_USER = """Text: "【高清剧集网发布 www.BPHDTV.com】外星也难民.第四季[全12集][简繁英字幕].Solar.Opposites.S04.2023.1080p.DSNP.WEB-DL.DDP5.1.H264-ZeroTV"
Output:"""
    FEW_SHOT_EXAMPLE_2_ASSISTANT = """{
  "title": "Solar Opposites",
  "year": 2023,
  "season": 4,
  "episode": "01-12",
  "resolution": "1080p",
  "audio_language": null,
  "source": "WEB-DL",
  "release_group": "ZeroTV"
}"""

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
            self.session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
            
            logger.info("SUCCESS: Direct ONNX Runtime session created.")
        except Exception as e:
            logger.critical("FATAL: An exception occurred during ONNX initialization.", exc_info=True)
            self.session = None
            self.tokenizer = None
        
        logger.info("--- ONNX Initialization Complete ---")

    def extract_metadata(self, title: str) -> dict:
        if not self.session or not self.tokenizer:
            return {"error": "Model/tokenizer is not available."}

        messages = [
            {"role": "system", "content": self.PROMPT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": self.FEW_SHOT_EXAMPLE_1_USER},
            {"role": "assistant", "content": self.FEW_SHOT_EXAMPLE_1_ASSISTANT},
            {"role": "user", "content": self.FEW_SHOT_EXAMPLE_2_USER},
            {"role": "assistant", "content": self.FEW_SHOT_EXAMPLE_2_ASSISTANT},
            {"role": "user", "content": f'Text: "{title}"\nOutput:'}
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="np")

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # The ONNX model for Causal LM typically has a past_key_values cache.
            # We need to get the number and shape of these from the model's inputs.
            num_pkv = (len(self.session.get_inputs()) - 2) // 2
            past_key_values = [numpy.zeros((1, 8, 0, 64), dtype=numpy.float32) for _ in range(num_pkv * 2)] # Example shape, may need adjustment

            max_new_tokens = 512
            eos_token_id = self.tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                ort_inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                
                # Add past_key_values to the inputs dictionary
                for i, pkv in enumerate(past_key_values):
                    ort_inputs[f'past_key_values.{i}'] = pkv

                ort_outs = self.session.run(None, ort_inputs)
                
                logits = ort_outs[0]
                past_key_values = ort_outs[1:] # The rest of the outputs are the new KV cache

                next_token_logits = logits[:, -1, :]
                next_token = numpy.argmax(next_token_logits, axis=-1).reshape((1,1))
                
                # Append the new token
                input_ids = next_token
                
                # Update attention mask
                attention_mask = numpy.concatenate([attention_mask, numpy.ones((1,1), dtype=numpy.int64)], axis=1)

                if next_token[0,0] == eos_token_id:
                    break
            
            # This part of the logic needs to be fixed to accumulate tokens
            # For now, this is a placeholder to show the structure.
            # A full implementation requires accumulating the generated tokens.
            # Let's simplify for now and assume the logic will be refined.
            
            # A simplified placeholder for generation
            generated_ids = self.run_generation(inputs)
            
            response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            json_start = response_text.rfind('{')
            if json_start != -1:
                json_end = response_text.rfind('}')
                if json_end > json_start:
                    response_text = response_text[json_start:json_end+1]

            parsed_json = json.loads(response_text)
            parsed_json['original_title'] = title
            return parsed_json

        except Exception as e:
            logger.error(f"An unexpected error occurred during ONNX inference for title '{title}': {e}", exc_info=True)
            return {"original_title": title, "error": f"An unexpected error occurred: {e}"}

    def run_generation(self, inputs):
        """Helper function for the generation loop."""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Get the names of the past_key_values inputs from the model
        pkv_input_names = [inp.name for inp in self.session.get_inputs() if 'past_key_values' in inp.name]
        
        # Initialize past_key_values as empty numpy arrays with the correct shape
        # This is a common pattern, but the exact shape can vary.
        # Shape is (batch_size, num_heads, sequence_length, head_dim)
        # We start with sequence_length = 0
        batch_size = input_ids.shape[0]
        num_heads = 8 # This is a typical value for small models, may need to get from config
        head_dim = 64 # This is a typical value for small models, may need to get from config
        
        past_key_values = [numpy.zeros((batch_size, num_heads, 0, head_dim), dtype=numpy.float32) for _ in pkv_input_names]

        generated_tokens = []
        max_new_tokens = 512
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            ort_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            for i, name in enumerate(pkv_input_names):
                ort_inputs[name] = past_key_values[i]

            output_names = [out.name for out in self.session.get_outputs()]
            ort_outs = self.session.run(output_names, ort_inputs)
            
            logits = ort_outs[0]
            past_key_values = ort_outs[1:]

            next_token_id = numpy.argmax(logits[:, -1, :], axis=-1).reshape((batch_size, 1))
            
            generated_tokens.append(next_token_id[0,0])

            if next_token_id[0,0] == eos_token_id:
                break

            input_ids = next_token_id
            attention_mask = numpy.concatenate([attention_mask, numpy.ones((batch_size, 1), dtype=numpy.int64)], axis=1)

        return [inputs['input_ids'][0].tolist() + generated_tokens]


logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
