from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fal_client
import os

app = FastAPI()

# Enable CORS so your Figma site can talk to Railway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your .figma.site URL for better security later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoSegmentRequest(BaseModel):
    image_url: str

@app.post("/segment")
async def segment_and_label(request: AutoSegmentRequest):
    print(f"Processing image: {request.image_url}")
    
    try:
        # We use Florence-2 for 'captioned' segmentation (labels + masks)
        result = fal_client.subscribe(
            "fal-ai/florence-2",
            arguments={
                "image_url": request.image_url,
                "task_prompt": "referring_expression_segmentation",
            }
        )

        # Florence-2 returns a list of detections with labels and polygon data
        # We transform it into a clean list for Figma
        raw_results = result.get("results", [])
        formatted_masks = []

        for item in raw_results:
            formatted_masks.append({
                "label": item.get("label", "Unknown Object"),
                "polygon": item.get("polygon", []), # Array of [x, y] points
            })

        print(f"Successfully found {len(formatted_masks)} objects.")
        return {"masks": formatted_masks}

    except Exception as e:
        print(f"Error calling Fal API: {str(e)}")
        return {"masks": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)