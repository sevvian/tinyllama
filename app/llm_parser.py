# R37 (NEW): The entire generation logic is rewritten to correctly handle the ONNX model's
# strict input requirements for past_key_values on the initial call.
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
CONFIG_PATH = os.path.join(LOCAL_MODEL_DIR, "config.json")

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
    FEW_SHOT_EXAMPLE_2_USER = """Text: "【高清剧集网发布 www.BPHDTV.com】外星也難民.第四季[全12集][簡繁英字幕].Solar.Opposites.S04.2023.1080p.DSNP.WEB-DL.DDP5.1.H264-ZeroTV"
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

            # R37: Load model configuration to get parameters for KV cache shape.
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            self.num_heads = config["num_attention_heads"]
            self.hidden_size = config["hidden_size"]
            self.head_dim = self.hidden_size // self.num_heads
            
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

    # R37: This function is completely rewritten to implement the correct two-phase generation.
    def run_generation(self, inputs):
        """Runs autoregressive generation with the ONNX model, correctly handling the KV cache."""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        batch_size, sequence_length = input_ids.shape

        # Get the names of all expected inputs from the ONNX model.
        input_names = [inp.name for inp in self.session.get_inputs()]
        pkv_input_names = [name for name in input_names if 'past_key_values' in name]
        output_names = [out.name for out in self.session.get_outputs()]

        # --- Phase 1: Process the entire prompt at once ---
        
        # Create the 'position_ids' for the initial prompt.
        position_ids = numpy.arange(sequence_length, dtype=numpy.int64).reshape(batch_size, sequence_length)

        # Create empty placeholder 'past_key_values' for the first run.
        # Shape is (batch_size, num_heads, sequence_length=0, head_dim).
        empty_past = [numpy.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=numpy.float32) for _ in pkv_input_names]

        ort_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        # Add the empty placeholders to the input feed.
        for i, name in enumerate(pkv_input_names):
            ort_inputs[name] = empty_past[i]

        # Run the first pass.
        ort_outs = self.session.run(output_names, ort_inputs)
        logits = ort_outs[0]
        past_key_values = ort_outs[1:] # This is now the populated KV cache for the prompt.

        # Get the very first generated token.
        next_token_id = numpy.argmax(logits[:, -1, :], axis=-1).reshape((batch_size, 1))
        generated_tokens = [next_token_id[0, 0]]

        # --- Phase 2: Generate subsequent tokens one by one ---

        max_new_tokens = 512
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            if generated_tokens[-1] == eos_token_id:
                break

            # For the next step, the input is just the new token.
            input_ids = next_token_id
            
            # The position is the length of the sequence so far.
            position_ids = numpy.array([[sequence_length]], dtype=numpy.int64)
            sequence_length += 1
            
            # The attention mask must be extended.
            attention_mask = numpy.concatenate([attention_mask, numpy.ones((batch_size, 1), dtype=numpy.int64)], axis=1)

            ort_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
            }
            # Add the KV cache from the *previous* step.
            for i, name in enumerate(pkv_input_names):
                ort_inputs[name] = past_key_values[i]
            
            # Run inference for the single new token.
            ort_outs = self.session.run(output_names, ort_inputs)
            logits = ort_outs[0]
            past_key_values = ort_outs[1:] # Update the cache for the next round.

            next_token_id = numpy.argmax(logits, axis=-1).reshape((batch_size, 1))
            generated_tokens.append(next_token_id[0, 0])

        return [inputs['input_ids'][0].tolist() + generated_tokens]


logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
