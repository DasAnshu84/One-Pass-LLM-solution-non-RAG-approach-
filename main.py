import os
import sys

# Force pure-Python implementation for protobuf to avoid Python 3.14 compatibility errors
sys.modules['google._upb._message'] = None
sys.modules['google.protobuf.pyext._message'] = None
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Load initial dotenv configuration
load_dotenv()

app = FastAPI(title="Gemini API Test UI")

# Setup CORS for development and cross-origin testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for generation requests
class GenerateRequest(BaseModel):
    prompt: str

@app.get("/api/status")
async def get_status():
    """
    Check if the Gemini API key is configured.
    Loads environment variables dynamically to pick up any changes without restart.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    configured = bool(api_key and api_key.strip() and api_key != "your_actual_api_key_here")
    
    if configured:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            configured = False
            
    return {"configured": configured}

@app.post("/api/generate")
async def generate_text(request: GenerateRequest):
    """
    Generate text using Gemini 1.5 Flash based on user prompt.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key or not api_key.strip() or api_key == "your_actual_api_key_here":
        raise HTTPException(
            status_code=400, 
            detail="Gemini API key is not configured. Please add your key to the .env file."
        )
    
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(
            status_code=400, 
            detail="Prompt cannot be empty."
        )
    
    try:
        # Re-configure to ensure we are using the current key
        genai.configure(api_key=api_key.strip())
        
        # Instantiate the model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Generate content
        response = model.generate_content(request.prompt)
        
        if not response or not response.text:
            raise HTTPException(status_code=500, detail="Received an empty response from Gemini.")
            
        return {"response": response.text}
    except Exception as e:
        # Log and return a clean error message
        error_msg = str(e)
        if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
            raise HTTPException(status_code=401, detail="The Gemini API key provided in the .env file is invalid.")
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {error_msg}")

# Ensure static directory exists (points to UI/static)
static_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "UI", "static"))

# Mount static folder if it exists
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_root():
    """
    Serve index.html at root route.
    """
    if not os.path.exists(static_dir):
        return {
            "error": "Static directory not found",
            "expected_path": static_dir,
            "message": "Please ensure the frontend files are in the UI/static folder."
        }
    
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "error": "index.html not found",
        "path": index_path,
        "message": "Please ensure index.html exists in the UI/static folder."
    }
