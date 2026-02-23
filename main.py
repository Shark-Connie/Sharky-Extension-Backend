import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sharky AI Extension API")

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}

config = load_config()

# Allow CORS so the browser extension can talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production for extension, this might need specific chrome-extension:// origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = ""
    # Add other parameters like negative_prompt, controlnet_image, etc., later.

@app.get("/")
def read_root():
    return {"message": "Sharky AI Backend is running."}

@app.get("/status")
def get_status():
    """Endpoint for the extension to check if the AI server is online."""
    return {
        "status": "online",
        "provider": config.get("ai_settings", {}).get("provider", "unknown")
    }

@app.post("/generate")
def generate_image(request: GenerateRequest):
    """Placeholder for eventual Stable Diffusion generation."""
    return {"status": "success", "message": "Image generation not yet implemented", "prompt_received": request.prompt}
