# R17, R18: This module now loads a local ONNX model that was exported during the build.
import json
import os
import logging
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoProcessor

# --- Basic Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

DEVICE = "cpu"
# R18: The MODEL_ID is now a local path inside the container.
LOCAL_MODEL_PATH = "./model"

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
        if hasattr(self, 'model'):
            return
            
        logger.info("--- Starting ONNX Model Initialization from Local Files ---")
        logger.info(f"Loading model and processor from path: {LOCAL_MODEL_PATH}")
        
        try:
            # Load the processor and model from the local directory baked into the image.
            self.processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH)
            self.model = ORTModelForVision2Seq.from_pretrained(LOCAL_MODEL_PATH, provider="CPUExecutionProvider")
            
            logger.info("SUCCESS: Local ONNX model loaded and ready.")
        except Exception as e:
            logger.critical("FATAL: An exception occurred during local ONNX model initialization.", exc_info=True)
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
            
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.debug(f"Raw model response text: {response_text}")

            json_start = response_text.rfind('{')
            if json_start != -1:
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
