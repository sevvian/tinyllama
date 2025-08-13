from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from llm_parser import parser_instance

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Metadata Extractor",
    description="An API to extract metadata from media titles using a SmolLM2 ONNX model.",
    version="3.0.0" # Final Version
)

logger.info("FastAPI app starting up...")

app.mount("/static", StaticFiles(directory="static"), name="static")

class ExtractionRequest(BaseModel):
    titles: str

class ExtractionResponse(BaseModel):
    results: List[Dict[str, Any]]

@app.post("/api/extract", response_model=ExtractionResponse)
def extract_metadata_api(request: ExtractionRequest):
    titles = [title.strip() for title in request.titles.split('\n') if title.strip()]
    logger.info(f"Received API request to process {len(titles)} titles.")
    results = [parser_instance.extract_metadata(title) for title in titles]
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

logger.info("FastAPI app definition complete.")
