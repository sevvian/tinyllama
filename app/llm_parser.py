# R1: This module contains the core logic for loading the LLM and parsing text.
# R6, R7: Enhanced with detailed, configurable logging.
import json
import os
import logging
from llama_cpp import Llama

# R7: Make logging level configurable via an environment variable.
# Defaults to 'INFO' if not set.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetadataParser:
    """
    A class to handle loading the LLM and extracting metadata from titles.
    The model is loaded only once upon instantiation to save resources.
    """
    _instance = None
    
    MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
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

Text: "【高清剧集网发布 www.BPHDTV.com】外星也难民.第四季[全12集][简繁英字幕].Solar.Opposites.S04.2023.1080p.DSNP.WEB-DL.DDP5.1.H24-ZeroTV"
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
            
        logger.info("Initializing and loading LLM model... This may take a moment.")
        try:
            self.llm = Llama(
              model_path=self.MODEL_PATH,
              n_ctx=2048,
              n_threads=4,
              n_gpu_layers=0,
              verbose=False # Turn off llama.cpp's native verbose logging to use our own
            )
            logger.info("LLM Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.llm = None

    def extract_metadata(self, title: str) -> dict:
        if not self.llm:
            return {"error": "LLM model is not available."}
        
        if not title or not title.strip():
            return {"original_title": title, "error": "Input title is empty."}

        # R6: Log the title being processed.
        logger.info(f"Processing title: '{title}'")
        
        prompt = self.PROMPT_TEMPLATE.format(title=title)
        
        # R7: Log the full prompt only at DEBUG level to avoid cluttering INFO logs.
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
            
            # R7: Log the raw LLM output at DEBUG level.
            logger.debug(f"Raw LLM response text: {response_text}")

            json_start = response_text.find('{')
            if json_start != -1:
                response_text = response_text[json_start:]
            
            parsed_json = json.loads(response_text)
            parsed_json['original_title'] = title

            # R6: Log the successful outcome.
            logger.info(f"Successfully parsed metadata for title: '{title}'")
            # R7: Log the actual parsed data at DEBUG level.
            logger.debug(f"Returning parsed JSON: {json.dumps(parsed_json)}")

            return parsed_json

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response for title '{title}'. Response was: {response_text}")
            return {"original_title": title, "error": "LLM returned malformed data."}
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM inference for title '{title}': {e}")
            return {"original_title": title, "error": f"An unexpected error occurred: {e}"}

parser_instance = MetadataParser()
