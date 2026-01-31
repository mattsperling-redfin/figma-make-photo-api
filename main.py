from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fal_client

app = FastAPI()

# CRITICAL: This allows your Figma site to talk to this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://desk-lively-90397360.figma.site"], # In production, replace "*" with your figma.site URL
    allow_methods=["*"],
    allow_headers=["*"],
)

class SegmentRequest(BaseModel):
    image_url: str
    x: int
    y: int

@app.post("/segment")
async def segment(request: SegmentRequest):
    # This calls the Fal SAM2 model
    handler = fal_client.submit(
        "fal-ai/sam2/image",
        arguments={
            "image_url": request.image_url,
            "prompts": [{"x": request.x, "y": request.y, "label": 1}]
        }
    )
    result = handler.get()
    return {"mask_url": result['image']['url']}