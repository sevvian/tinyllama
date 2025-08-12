# R1: Defines the FastAPI application and API endpoints.
# R6: Added top-level logging for API requests.
# R8: Fixed import error by changing relative import to absolute.
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

# R8: FIX - Changed from ".llm_parser" to "llm_parser".
# This is now an absolute import that works correctly when uvicorn runs main.py.
from llm_parser import parser_instance # Import the singleton instance

# Get the logger instance configured in llm_parser
logger = logging.getLogger(__name__)

# Define the application
app = FastAPI(
    title="LLM Metadata Extractor",
    description="An API to extract metadata from media titles using a TinyLLM model.",
    version="1.0.0"
)

# Mount the 'static' directory to serve frontend files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Pydantic models for request and response validation
class ExtractionRequest(BaseModel):
    titles: str # A single string which can contain multiple titles separated by newlines

class ExtractionResponse(BaseModel):
    results: List[Dict[str, Any]]

# R1: API endpoint for metadata extraction
@app.post("/api/extract", response_model=ExtractionResponse)
def extract_metadata_api(request: ExtractionRequest):
    """
    Accepts a block of text with one title per line and returns extracted metadata for each.
    """
    titles = [title.strip() for title in request.titles.split('\n') if title.strip()]
    
    # R6: Log the incoming request details.
    logger.info(f"Received API request to process {len(titles)} titles.")
    
    results = [parser_instance.extract_metadata(title) for title in titles]
    return {"results": results}

# R2: Root endpoint to serve the frontend UI
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    """
    Serves the main HTML page for the user interface.
    """
    with open("app/static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)
