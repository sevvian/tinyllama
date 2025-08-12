# R12: Switched to the Qwen3-0.6B model and updated prompt/stop tokens accordingly.
import json
import os
import logging
from llama_cpp import Llama

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class MetadataParser:
    _instance = None
    
    # R12: FIX - Point to the new Qwen3 GGUF file. The filename is exact.
    MODEL_PATH = "./models/unsloth_qwen3-0.6b-q4_k_s.gguf"
    
    # R12: FIX - Update the prompt template to the "Qwen Chat" format.
    # It uses <|im_start|> and <|im_end|> tokens.
    PROMPT_TEMPLATE = """<|im_start|>system
You are an expert file name parser. Your task is to extract metadata from the user's text and return a clean JSON object. The fields to extract are: title, year, season, episode, resolution, audio_language, source, release_group. If a field is not present, return it as null. Respond with ONLY the JSON object.<|im_end|>
<|im_start|>user
Text: "www.Tamilblasters.qpon - Alice In Borderland (2020) S02 EP (01-08) - HQ HDRip - 720p - [Tam+ Hin + Eng] - (AAC 2.0) - 2.8GB - ESub"
Output:
{{
  "title": "Alice In Borderland",
  "year": 2020,
  "season": 2,
  "episode": "01-08",
  "resolution": "720p",
  "audio_language": ["Tamil", "Hindi", "English"],
  "source": "HDRip",
  "release_group": null
}}

Text: "【高清剧集网发布 www.BPHDTV.com】外星也难民.第四季[全12集][简繁英字幕].Solar.Opposites.S04.2023.1080p.DSNP.WEB-DL.DDP5.1.H264-ZeroTV"
Output:
{{
  "title": "Solar Opposites",
  "year": 2023,
  "season": 4,
  "episode": "01-12",
  "resolution": "1080p",
  "audio_language": null,
  "source": "WEB-DL",
  "release_group": "ZeroTV"
}}

Now, process the following text:
Text: "{title}"
Output:<|im_end|>
<|im_start|>assistant
"""

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MetadataParser, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'llm'):
            return
            
        logger.info("--- Starting LLM Initialization ---")
        logger.info(f"Verifying model file at path: {self.MODEL_PATH}")
        if os.path.exists(self.MODEL_PATH):
            logger.info("SUCCESS: Model file found.")
        else:
            logger.critical("FATAL: Model file NOT FOUND at the specified path.")
            self.llm = None
            return

        logger.info("Attempting to instantiate Llama model...")
        try:
            self.llm = Llama(
              model_path=self.MODEL_PATH,
              n_ctx=2048,
              n_threads=4,
              n_gpu_layers=0,
              verbose=False
            )
            logger.info("SUCCESS: Llama model instantiated successfully.")
        except Exception as e:
            logger.critical("FATAL: An exception occurred during Llama model instantiation.", exc_info=True)
            self.llm = None
            return
        logger.info("--- LLM Initialization Complete ---")

    def extract_metadata(self, title: str) -> dict:
        if not self.llm:
            logger.error(f"Cannot process '{title}' because LLM is not available.")
            return {"error": "LLM model is not available."}
        
        if not title or not title.strip():
            return {"original_title": title, "error": "Input title is empty."}

        logger.info(f"Processing title: '{title}'")
        prompt = self.PROMPT_TEMPLATE.format(title=title)
        logger.debug(f"Generated prompt for LLM:\n{prompt}")

        try:
            output = self.llm(
              prompt,
              max_tokens=512,
              # R12: FIX - Update stop tokens for the Qwen chat template.
              stop=["<|im_end|>", "}"],
              temperature=0.2,
              echo=False
            )
            
            response_text = output['choices'][0]['text'].strip()
            if not response_text.endswith('}'):
                response_text += "}"
            
            logger.debug(f"Raw LLM response text: {response_text}")

            json_start = response_text.find('{')
            if json_start != -1:
                response_text = response_text[json_start:]
            
            parsed_json = json.loads(response_text)
            parsed_json['original_title'] = title

            logger.info(f"Successfully parsed metadata for title: '{title}'")
            logger.debug(f"Returning parsed JSON: {json.dumps(parsed_json)}")

            return parsed_json

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response for title '{title}'. Response was: {response_text}")
            return {"original_title": title, "error": "LLM returned malformed data."}
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM inference for title '{title}': {e}", exc_info=True)
            return {"original_title": title, "error": f"An unexpected error occurred: {e}"}

logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
