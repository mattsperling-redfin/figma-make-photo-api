from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fal_client
import os

app = FastAPI()

# Enable CORS for Figma
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoSegmentRequest(BaseModel):
    image_url: str

@app.post("/segment")
async def auto_segment(request: AutoSegmentRequest):
    print(f"🚀 Starting segmentation for: {request.image_url}")
    
    try:
        # Standard SAM 2 Auto-Segment Endpoint
        result = fal_client.subscribe(
            "fal-ai/sam2/auto-segment",
            arguments={"image_url": request.image_url}
        )
        
        # LOGGING: This helps you see the actual structure in Railway logs
        print(f"DEBUG: Full Fal Result keys: {result.keys()}")

        # Fal usually returns 'individual_masks' for the array of objects
        # We check multiple keys just in case the API updates
        masks_data = result.get("individual_masks") or result.get("masks") or result.get("segments") or []
        
        # If it's still empty, it might be a 'combined_mask' only, 
        # but auto-segment should provide the list.
        if not masks_data:
            print("⚠️ No individual masks found in response.")

        return {"masks": masks_data}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"masks": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)