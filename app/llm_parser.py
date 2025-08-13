# This is the final, definitive version of the parser.
# It reverts to the simple, effective prompt and generation logic from the
# known working version ("R38"), while keeping the robust loading and parsing improvements.
import json
import os
import logging
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
import numpy

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR = "./model"
TOKENIZER_PATH = LOCAL_MODEL_DIR
ONNX_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.onnx")
CONFIG_PATH = os.path.join(LOCAL_MODEL_DIR, "config.json")

class MetadataParser:
    _instance = None
    
    # R53: Reverted to the simple, effective prompt from the working version.
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
    FEW_SHOT_EXAMPLE_2_USER = """Text: "[apreder]Raffaello_Il_Principe_delle_Arti_in_3D(2017)DVB.mkv"
Output:"""
    FEW_SHOT_EXAMPLE_2_ASSISTANT = """{
  "title": "Raffaello Il Principe delle Arti in 3D",
  "year": 2017,
  "season": null,
  "episode": null,
  "resolution": null,
  "audio_language": null,
  "source": "DVB",
  "release_group": "apreder"
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
            
            # Keep the improved AutoConfig loader.
            logger.info(f"Loading model config from path: {CONFIG_PATH}")
            config = AutoConfig.from_pretrained(CONFIG_PATH)
            self.num_heads = config.num_key_value_heads 
            self.hidden_size = config.hidden_size
            self.head_dim = self.hidden_size // config.num_attention_heads
            
            logger.info(f"Loading ONNX model and creating inference session from: {ONNX_MODEL_PATH}")
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options=sess_options, providers=["CPUExecutionProvider"])
            
            logger.info(f"Model config loaded: num_heads for KV cache = {self.num_heads}, head_dim = {self.head_dim}")
            logger.info("SUCCESS: Direct ONNX Runtime session created.")
        except Exception as e:
            logger.critical("FATAL: An exception occurred during ONNX initialization.", exc_info=True)
            self.session = None
            self.tokenizer = None
        
        logger.info("--- ONNX Initialization Complete ---")

    def extract_metadata(self, title: str) -> dict:
        if not self.session or not self.tokenizer:
            return {"error": "Model/tokenizer is not available."}

        # Use the simple, effective two-shot prompt.
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
            
            prompt_len = inputs['input_ids'].shape[1]
            newly_generated_ids = generated_ids[0][prompt_len:]
            
            response_text = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
            logger.debug(f"Cleaned model response text: {response_text}")
            
            # Keep the improved, robust JSON parsing.
            clean_response = response_text.strip()
            if clean_response and clean_response.startswith('{') and clean_response.endswith('}'):
                parsed_json = json.loads(clean_response)
                parsed_json['original_title'] = title
                return parsed_json
            else:
                logger.error(f"Model returned a non-JSON response: '{response_text}'")
                return {"original_title": title, "error": "Model failed to generate valid JSON."}

        except Exception as e:
            logger.error(f"An unexpected error occurred during ONNX inference for title '{title}': {e}", exc_info=True)
            return {"original_title": title, "error": f"An unexpected error occurred: {e}"}

    # R53: Reverted to the simple, deterministic generation logic from the working version.
    def run_generation(self, inputs):
        """Runs autoregressive generation with the ONNX model, correctly handling the KV cache."""
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        batch_size, sequence_length = input_ids.shape

        input_names = [inp.name for inp in self.session.get_inputs()]
        pkv_input_names = [name for name in input_names if 'past_key_values' in name]
        output_names = [out.name for out in self.session.get_outputs()]

        position_ids = numpy.arange(sequence_length, dtype=numpy.int64).reshape(batch_size, sequence_length)
        empty_past = [numpy.zeros((batch_size, self.num_heads, 0, self.head_dim), dtype=numpy.float32) for _ in pkv_input_names]
        ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
        for i, name in enumerate(pkv_input_names):
            ort_inputs[name] = empty_past[i]
        ort_outs = self.session.run(output_names, ort_inputs)
        logits = ort_outs[0]
        past_key_values = ort_outs[1:]
        
        next_token_id = numpy.argmax(logits[:, -1, :], axis=-1).reshape((batch_size, 1))
        generated_tokens = [next_token_id[0, 0]]
        
        max_new_tokens = 512
        eos_token_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens - 1):
            if generated_tokens[-1] == eos_token_id:
                break

            input_ids = next_token_id
            position_ids = numpy.array([[sequence_length]], dtype=numpy.int64)
            sequence_length += 1
            attention_mask = numpy.concatenate([attention_mask, numpy.ones((batch_size, 1), dtype=numpy.int64)], axis=1)

            ort_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
            for i, name in enumerate(pkv_input_names):
                ort_inputs[name] = past_key_values[i]
            
            ort_outs = self.session.run(output_names, ort_inputs)
            logits = ort_outs[0]
            past_key_values = ort_outs[1:]
            
            next_token_id = numpy.argmax(logits, axis=-1).reshape((batch_size, 1))
            
            generated_tokens.append(next_token_id[0, 0])

        return [inputs['input_ids'][0].tolist() + generated_tokens]


logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
