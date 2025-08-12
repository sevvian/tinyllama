# R1: This module contains the core logic for loading the LLM and parsing text.
import json
from llama_cpp import Llama
from pydantic import BaseModel, ValidationError
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataParser:
    """
    A class to handle loading the LLM and extracting metadata from titles.
    The model is loaded only once upon instantiation to save resources.
    """
    _instance = None
    
    # R1: The model path is fixed as it's defined in the Dockerfile build process.
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
        # This check prevents re-initialization on subsequent calls
        if hasattr(self, 'llm'):
            return
            
        logger.info("Initializing and loading LLM model... This may take a moment.")
        try:
            self.llm = Llama(
              model_path=self.MODEL_PATH,
              n_ctx=2048,      # Context window size
              n_threads=4,     # Number of CPU threads to use
              n_gpu_layers=0   # Explicitly set to 0 for CPU-only inference
            )
            logger.info("LLM Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self.llm = None

    def extract_metadata(self, title: str) -> dict:
        """
        Takes a single title string and returns a dictionary of extracted metadata.
        """
        if not self.llm:
            return {"error": "LLM model is not available."}
        
        if not title or not title.strip():
            return {"original_title": title, "error": "Input title is empty."}

        prompt = self.PROMPT_TEMPLATE.format(title=title)

        try:
            output = self.llm(
              prompt,
              max_tokens=512,
              stop=["</s>", "}"], # Stop generation after the JSON is complete
              temperature=0.2, # Lower temperature for more deterministic output
              echo=False
            )
            
            # The model's response is in the 'choices'['text'] field
            # We add the closing brace because we often stop generation on it
            response_text = output['choices']['text'].strip()
            if not response_text.endswith('}'):
                response_text += "}"
            
            # The model might sometimes add introductory text, we find the JSON object.
            json_start = response_text.find('{')
            if json_start != -1:
                response_text = response_text[json_start:]
            
            # Parse the JSON string into a Python dictionary
            parsed_json = json.loads(response_text)
            parsed_json['original_title'] = title # Add original title for reference
            return parsed_json

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM response: {response_text}")
            return {"original_title": title, "error": "LLM returned malformed data."}
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM inference: {e}")
            return {"original_title": title, "error": f"An unexpected error occurred: {e}"}

# Create a single instance to be used by the FastAPI app
parser_instance = MetadataParser()
