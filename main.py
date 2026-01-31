from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fal_client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Keep as * during testing
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoSegmentRequest(BaseModel):
    image_url: str

@app.post("/segment")
async def auto_segment(request: AutoSegmentRequest):
    # Calling the auto-segment model
    result = fal_client.subscribe(
        "fal-ai/sam2/auto-segment",
        arguments={
            "image_url": request.image_url,
        }
    )
    
    # We want to return the 'masks' array. 
    # Each mask contains a 'data' field or 'polygon' field depending on the model version.
    # For Figma, we will pass the list of mask objects.
    return {"masks": result.get("masks", [])}