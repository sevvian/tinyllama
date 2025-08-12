# R1, R6, R7: Core logic with configurable logging.
# R10: Reverted to TinyLlama model and Zephyr prompt.
# R11: Added detailed diagnostic logging for model loading.
import json
import os
import logging
from llama_cpp import Llama

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# Set a detailed format for logs to be more informative.
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class MetadataParser:
    """
    A class to handle loading the LLM and extracting metadata from titles.
    """
    _instance = None
    
    # R10: Reverted to the TinyLlama model path.
    MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # R10: Reverted to the Zephyr prompt template for TinyLlama.
    PROMPT_TEMPLATE = """<|system|>
You are an expert file name parser. Your task is to extract metadata from the user's text and return a clean JSON object. The fields to extract are: title, year, season, episode, resolution, audio_language, source, release_group. If a field is not present, return it as null. Respond with ONLY the JSON object.</s>
<|user|>
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
Output:
</s>
<|assistant|>
"""

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MetadataParser, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'llm'):
            return
        
        # R11: Add detailed diagnostic steps.
        logger.info("--- Starting LLM Initialization ---")
        
        # Step 1: Verify the model file exists at the expected path.
        logger.info(f"Verifying model file at path: {self.MODEL_PATH}")
        if os.path.exists(self.MODEL_PATH):
            logger.info("SUCCESS: Model file found.")
        else:
            logger.critical("FATAL: Model file NOT FOUND at the specified path.")
            # Let's also log the contents of the /app and /app/models directories for debugging.
            try:
                app_dir_contents = os.listdir("/app")
                models_dir_contents = os.listdir("/app/models")
                logger.debug(f"Contents of /app: {app_dir_contents}")
                logger.debug(f"Contents of /app/models: {models_dir_contents}")
            except FileNotFoundError as e:
                logger.error(f"Could not list directory contents: {e}")
            self.llm = None
            return

        # Step 2: Attempt to load the model with clear start/end logs.
        logger.info("Attempting to instantiate Llama model...")
        try:
            self.llm = Llama(
              model_path=self.MODEL_PATH,
              n_ctx=2048,
              n_threads=4,
              n_gpu_layers=0,
              verbose=False # We use our own logging.
            )
            logger.info("SUCCESS: Llama model instantiated successfully.")
        except Exception as e:
            # R11: Log the full exception with traceback to get the exact error.
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
              stop=["</s>", "}"],
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
