import json
import os

# Isolate AI models cache to project folder
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)
os.environ["HF_HOME"] = models_dir
os.environ["XDG_CACHE_HOME"] = models_dir
os.environ["TORCH_HOME"] = models_dir

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
from PIL import Image, ImageDraw
try:
    from simple_lama_inpainting import SimpleLama
    print("Loading LaMa model... This may take a minute on first run.")
    lama = SimpleLama()
    print("LaMa model loaded successfully.")
except ImportError:
    print("Failed to load SimpleLama. Make sure it is installed.")
    lama = None

try:
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='auto', target='en')
    print("GoogleTranslator loaded successfully for AI prompts.")
except ImportError:
    print("Failed to load deep_translator. Prompts will not be translated.")
    translator = None

try:
    import torch
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    from diffusers import AutoPipelineForInpainting
    from diffusers.utils import logging as diffusers_logging
    
    # Suppress the long safety checker warning
    diffusers_logging.set_verbosity_error()
    
    print("Loading Stable Diffusion Inpainting model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    sd_pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch_dtype,
        variant="fp16" if device == "cuda" else None,
        safety_checker=None
    )
    sd_pipe.to(device)
    print(f"Stable Diffusion loaded successfully on {device}")
except ImportError:
    print("Failed to load diffusers. Stable Diffusion Inpainting disabled.")
    sd_pipe = None
except Exception as e:
    print(f"Failed to load SD model: {e}")
    sd_pipe = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print("Loading BLIP Vision Model for Image Context...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=models_dir)
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=models_dir)
    # BLIP is small enough to load in fp16 to save memory
    if device == "cuda":
        blip_model = blip_model.to(torch.float16)
    blip_model.to(device)
    print(f"BLIP Model loaded successfully on {device}.")
except ImportError:
    print("Transformers library not found. BLIP Context will be disabled.")
    blip_processor = None
    blip_model = None
except Exception as e:
    print(f"Failed to load BLIP Model: {e}")
    blip_processor = None
    blip_model = None

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    
    print("Loading Real-ESRGAN model (x2plus for speed)...")
    # Use 2x upscaler model for significantly faster inference
    realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        dni_weight=None,
        model=realesrgan_model,
        tile=400, # Use tiling to save VRAM on large images
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False,
        gpu_id=0 if torch.cuda.is_available() else None
    )
    print("Real-ESRGAN loaded successfully.")
except ImportError:
    print("Failed to load Real-ESRGAN. Enhancement disabled. Make sure target packages are installed.")
    upsampler = None
except Exception as e:
    print(f"Failed to load Real-ESRGAN model: {e}")
    upsampler = None

app = FastAPI(title="Sharky AI Extension API")

# Global state for generation progress
generation_progress = {
    "status": "idle",
    "step": 0,
    "total": 0
}

@app.get("/status")
def server_status():
    return {"status": "ok", "message": "Sharky AI Backend is running."}

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down AI Server gracefully...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
    except Exception as e:
        print(f"Error clearing cache: {e}")

@app.get("/progress")
async def get_progress():
    return generation_progress

def sd_progress_callback(pipe, step_index, timestep, callback_kwargs):
    generation_progress["step"] = step_index + 1
    return callback_kwargs

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

class InpaintRect(BaseModel):
    x: int
    y: int
    w: int
    h: int

class InpaintRequest(BaseModel):
    image: str
    rects: List[InpaintRect]
    prompt: str = ""

@app.post("/inpaint")
def inpaint_image(request: InpaintRequest):
    if lama is None and sd_pipe is None:
        return {"status": "error", "message": "No AI models loaded (LaMa or SD) on the server."}
    
    try:
        # Decode base64 to PIL Image
        encoded_data = request.image.split(',')[1] if ',' in request.image else request.image
        img_data = base64.b64decode(encoded_data)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')

        # Create mask
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        for r in request.rects:
            draw.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], fill=255)

        # Handle Prompt Translation (if any text was provided, translate to English for AI)
        final_prompt = request.prompt
        if final_prompt and translator:
            try:
                final_prompt = translator.translate(final_prompt)
                print(f"Translated prompt for AI: '{request.prompt}' -> '{final_prompt}'")
            except Exception as e:
                print(f"Translation failed: {e}")

        # Choose AI Model based on prompt presence
        if final_prompt and final_prompt.strip() != "" and sd_pipe is not None:
            print(f"Using Stable Diffusion with prompt: {final_prompt}")
            
            # --- High-Res Inpainting: Crop to Bounding Box ---
            # Find bounding box of masks
            min_x, min_y = img.size[0], img.size[1]
            max_x, max_y = 0, 0
            for r in request.rects:
                if r.x < min_x: min_x = r.x
                if r.y < min_y: min_y = r.y
                if r.x + r.w > max_x: max_x = r.x + r.w
                if r.y + r.h > max_y: max_y = r.y + r.h
                
            # Add context padding around the mask so AI understands the background (like face angle/lighting)
            # Use at least 256 pixels, or 40% of the box size (whichever is larger) to ensure enough context
            crop_w = max_x - min_x
            crop_h = max_y - min_y
            padding = max(384, int(max(crop_w, crop_h) * 0.8))
            
            crop_box = (
                max(0, min_x - padding),
                max(0, min_y - padding),
                min(img.size[0], max_x + padding),
                min(img.size[1], max_y + padding)
            )
            
            infer_img = img.crop(crop_box)
            infer_mask = mask.crop(crop_box)
            crop_size = infer_img.size
            
            # Downscale cropped region if it's too large for SD (e.g. > 1024)
            max_dim = 1024
            needs_resize = crop_size[0] > max_dim or crop_size[1] > max_dim
            if needs_resize:
                ratio = min(max_dim / crop_size[0], max_dim / crop_size[1])
                new_w = int(crop_size[0] * ratio)
                new_h = int(crop_size[1] * ratio)
                # SD requires dimensions to be multiples of 8
                new_w = (new_w // 8) * 8
                new_h = (new_h // 8) * 8
                
                print(f"Downscaling crop from {crop_size} to ({new_w}, {new_h}) for SD Generation")
                infer_img = infer_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                infer_mask = infer_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
            else:
                # Still ensure dimensions are multiples of 8
                new_w = (crop_size[0] // 8) * 8
                new_h = (crop_size[1] // 8) * 8
                if new_w != crop_size[0] or new_h != crop_size[1]:
                    infer_img = infer_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    infer_mask = infer_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
            
            import time
            start_t = time.time()
            
            # --- Image Context & Prompt Enhancement ---
            num_steps = 45
            
            # Use BLIP to get image context if active
            image_context = ""
            if blip_processor and blip_model:
                try:
                    # Resize for fast inference (max 512 on one side)
                    b_ratio = min(512 / img.size[0], 512 / img.size[1])
                    if b_ratio < 1.0:
                        b_img = img.resize((int(img.size[0] * b_ratio), int(img.size[1] * b_ratio)), Image.Resampling.LANCZOS)
                    else:
                        b_img = img.copy()
                        
                    blip_inputs = blip_processor(b_img, return_tensors="pt").to(blip_model.device)
                    # BLIP is small enough to load in fp16, we cast the pixel values to match model dtype
                    blip_inputs["pixel_values"] = blip_inputs["pixel_values"].to(blip_model.dtype)
                    
                    blip_out = blip_model.generate(**blip_inputs)
                    image_context = blip_processor.decode(blip_out[0], skip_special_tokens=True)
                    print(f"[Vision AI] Detected Image Context: '{image_context}'")
                except Exception as e:
                    print(f"BLIP Context Failed: {e}")
            
            # Combine user prompt with the Vision AI's understanding of the scene
            if image_context:
                enhanced_prompt = f"a person wearing {final_prompt}, {final_prompt}, matching the scene of {image_context}, highly detailed, photorealistic, 8k resolution, natural lighting, masterpiece, sharp focus"
            else:
                enhanced_prompt = f"a person wearing {final_prompt}, {final_prompt}, highly detailed, photorealistic, 8k resolution, natural lighting, masterpiece, sharp focus"
            
            neg_prompt = "ugly, deformed, disfigured, poor details, bad anatomy, blur, cartoon, illustration, weird blending, out of frame, artificial, changing facial structure, changing eyes"
            
            generation_progress["total"] = num_steps
            generation_progress["step"] = 0
            generation_progress["status"] = "generating"
            
            try:
                sd_result = sd_pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=neg_prompt,
                    image=infer_img.convert("RGB"),
                    mask_image=infer_mask,
                    num_inference_steps=num_steps,
                    guidance_scale=12.0, # High guidance to force prompt compliance (pink heart-shaped)
                    strength=0.90,       # Retain 10% of underlying eyes to make it blend and keep identity naturally
                    callback_on_step_end=sd_progress_callback
                ).images[0]
            finally:
                generation_progress["status"] = "idle"
                
            print(f"Generation took {time.time() - start_t:.2f} seconds.")
            
            # Upscale back to cropped size if we altered it
            if sd_result.size != crop_size:
               sd_result = sd_result.resize(crop_size, Image.Resampling.LANCZOS)
               
            # Paste the generated crop back onto the original high-res image
            # using the local mask with gaussian blur to blend edges
            from PIL import ImageFilter
            soft_mask = mask.crop(crop_box).filter(ImageFilter.GaussianBlur(radius=4))
            result_img = img.copy()
            result_img.paste(sd_result, (crop_box[0], crop_box[1]), soft_mask)

        else:
            # Inpaint using LaMa (Object removal)
            if lama is None:
               raise Exception("LaMa model is not loaded on the server and no prompt was provided.")
            print("Using LaMa for Object Removal")
            result_img = lama(img, mask)

        # Encode back to base64
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        res_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return {"status": "success", "image": "data:image/png;base64," + res_b64}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

class EnhanceRequest(BaseModel):
    image: str
    target_width: Optional[int] = None
    target_height: Optional[int] = None

@app.post("/enhance")
def enhance_image(request: EnhanceRequest):
    if upsampler is None:
        return {"status": "error", "message": "Real-ESRGAN model is not loaded on the server."}
    
    try:
        import numpy as np
        import cv2
        
        # Decode base64 
        encoded_data = request.image.split(',')[1] if ',' in request.image else request.image
        img_data = base64.b64decode(encoded_data)
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Scale it down slightly if it's super huge to save on compute before enhancement
        if img.shape[1] > 1920 or img.shape[0] > 1920:
            scale_factor = min(1920 / img.shape[1], 1920 / img.shape[0])
            new_w = int(img.shape[1] * scale_factor)
            new_h = int(img.shape[0] * scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        print("Enhancing image using Real-ESRGAN (x2)...")
        import time
        start_t = time.time()
        
        # We only want to 'enhance' clarity without drastically multiplying resolution.
        # Run Real-ESRGAN x2 model without outscaling
        output, _ = upsampler.enhance(img, outscale=1)
        
        # Ensure the final output matches the user's requested dimensions perfectly
        if request.target_width is not None and request.target_height is not None:
            output = cv2.resize(output, (request.target_width, request.target_height), interpolation=cv2.INTER_LANCZOS4)
            
        print(f"Enhancement took {time.time() - start_t:.2f} seconds.")
        
        # Encode back to base64
        _, buffer = cv2.imencode('.png', output)
        res_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"status": "success", "image": "data:image/png;base64," + res_b64}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
