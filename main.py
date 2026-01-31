from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fal_client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoSegmentRequest(BaseModel):
    image_url: str

@app.post("/segment")
async def auto_segment(request: AutoSegmentRequest):
    # Updated to use the auto-segment model
    result = fal_client.subscribe(
        "fal-ai/sam2/auto-segment",
        arguments={
            "image_url": request.image_url,
        }
    )
    
    # fal-ai/sam2/auto-segment returns a 'combined_mask' image
    return {"mask_url": result['combined_mask']['url']}