# R17: This module is completely rewritten to use the Transformers library with ONNX Runtime.
import json
import os
import logging
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- Basic Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Use CPU for inference as required.
DEVICE = "cpu"
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

class MetadataParser:
    """
    A class to handle loading the SmolVLM ONNX model and extracting metadata.
    """
    _instance = None
    
    # The prompt template is now defined by the model's chat template logic.
    # We construct the messages list and let processor.apply_chat_template handle it.
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
        if hasattr(self, 'model'):
            return
            
        logger.info("--- Starting ONNX Model Initialization ---")
        logger.info(f"Loading model '{MODEL_ID}' for CPU inference.")
        
        try:
            # Load the processor which handles tokenization and prompt formatting
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            logger.info("Processor loaded successfully.")

            # Load the model, specifying that we want to use the ONNX version.
            # The `trust_remote_code=True` flag might be needed for some custom models.
            self.model = AutoModelForVision2Seq.from_pretrained(
                MODEL_ID,
                from_onnx=True,
                trust_remote_code=True,
            ).to(DEVICE)
            
            logger.info("SUCCESS: ONNX model loaded and ready.")
        except Exception as e:
            logger.critical("FATAL: An exception occurred during ONNX model initialization.", exc_info=True)
            self.model = None
            self.processor = None
        
        logger.info("--- ONNX Initialization Complete ---")

    def extract_metadata(self, title: str) -> dict:
        if not self.model or not self.processor:
            logger.error(f"Cannot process '{title}' because the model/processor is not available.")
            return {"error": "Model/processor is not available."}
        
        if not title or not title.strip():
            return {"original_title": title, "error": "Input title is empty."}

        logger.info(f"Processing title: '{title}'")

        # Construct the prompt using the model's required chat format.
        # This is a text-only prompt, which the model should handle.
        messages = [
            {"role": "system", "content": self.PROMPT_SYSTEM_INSTRUCTION},
            {"role": "user", "content": self.FEW_SHOT_EXAMPLE_1_USER},
            {"role": "assistant", "content": self.FEW_SHOT_EXAMPLE_1_ASSISTANT},
            {"role": "user", "content": self.FEW_SHOT_EXAMPLE_2_USER},
            {"role": "assistant", "content": self.FEW_SHOT_EXAMPLE_2_ASSISTANT},
            {"role": "user", "content": f'Text: "{title}"\nOutput:'}
        ]

        try:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, return_tensors="pt").to(DEVICE)

            logger.debug("Generating response from ONNX model...")
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, eos_token_id=self.processor.tokenizer.eos_token_id)
            
            # Decode the output, skipping special tokens
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.debug(f"Raw model response text: {response_text}")

            # The model might include the full chat history in its response, so we find the final JSON.
            json_start = response_text.rfind('{')
            if json_start != -1:
                # Find the corresponding closing brace
                json_end = response_text.rfind('}')
                if json_end > json_start:
                    response_text = response_text[json_start:json_end+1]

            parsed_json = json.loads(response_text)
            parsed_json['original_title'] = title

            logger.info(f"Successfully parsed metadata for title: '{title}'")
            return parsed_json

        except Exception as e:
            logger.error(f"An unexpected error occurred during ONNX inference for title '{title}': {e}", exc_info=True)
            return {"original_title": title, "error": f"An unexpected error occurred: {e}"}

logger.info("Creating MetadataParser instance...")
parser_instance = MetadataParser()
logger.info("MetadataParser instance created.")
