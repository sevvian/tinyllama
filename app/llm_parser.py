# R35: This module uses the onnxruntime library directly.
# R36 (NEW): Corrected the manual generation loop to provide the required 'position_ids'
# and correctly formatted 'past_key_values' inputs to the ONNX model.
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
            
            generated_ids = self.run_generation(inputs)
            
            response_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.debug(f"Raw model response text: {response_text}")
            
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

    # R36: This entire function is rewritten to be correct.
    def run_generation(self, inputs):
        """Helper function for the generation loop that correctly handles the ONNX model's specific inputs."""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        batch_size, sequence_length = input_ids.shape

        # R36: FIX - Create the mandatory 'position_ids' input.
        position_ids = numpy.arange(sequence_length, dtype=numpy.int64).reshape(1, sequence_length)

        # Get the names of all expected inputs, including the split KV cache.
        input_names = [inp.name for inp in self.session.get_inputs()]
        pkv_input_names = [name for name in input_names if 'past_key_values' in name]
        
        # Initialize the past_key_values as empty. This is the first pass.
        past_key_values = None
        
        generated_tokens = []
        max_new_tokens = 512
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            ort_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            }
            
            # R36: FIX - Correctly handle the past_key_values format.
            if past_key_values is not None:
                # On subsequent runs, we provide the KV cache from the previous step.
                for i, name in enumerate(pkv_input_names):
                    ort_inputs[name] = past_key_values[i]

            output_names = [out.name for out in self.session.get_outputs()]
            ort_outs = self.session.run(output_names, ort_inputs)
            
            logits = ort_outs[0]
            # The new KV cache is the rest of the outputs.
            past_key_values = ort_outs[1:]

            # Get the next token by finding the highest logit.
            next_token_id = numpy.argmax(logits[:, -1, :], axis=-1).reshape((batch_size, 1))
            generated_tokens.append(next_token_id[0,0])

            if next_token_id[0,0] == eos_token_id:
                break

            # Prepare inputs for the *next* iteration.
            input_ids = next_token_id
            # Update attention mask to include the new token.
            attention_mask = numpy.concatenate([attention_mask, numpy.ones((batch_size, 1), dtype=numpy.int64)], axis=1)
            # R36: FIX - The position_id for the next token is just the current sequence length.
            position_ids = numpy.array([[attention_mask.shape[1] - 1]], dtype=numpy.int64)

        return [inputs['input_ids'][0].tolist() + generated_tokens]


logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
