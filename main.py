import os
import cv2
import numpy as np
import requests
import fal_client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# FIX: This explicitly tells the browser to allow Figma to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Wildcard allows testing from any Figma project URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoSegmentRequest(BaseModel):
    image_url: str

def get_polygon_from_mask_url(url):
    resp = requests.get(url)
    nparr = np.frombuffer(resp.content, np.uint8)
    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # FIXED: Changed find_contours to findContours
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = approx.reshape(-1, 2).tolist()
        polygons.append(points)
    return polygons

@app.post("/segment")
async def segment_to_svg(request: AutoSegmentRequest):
    print(f"🚀 Processing: {request.image_url}")
    try:
        # Call the auto-segment model
        result = fal_client.subscribe(
            "fal-ai/sam2/auto-segment",
            arguments={"image_url": request.image_url}
        )
        
        # KEY FIX: Fal uses 'individual_masks' for the list of objects
        individual_masks = result.get("individual_masks") or []
        svg_data = []
        
        for idx, mask_obj in enumerate(individual_masks):
            mask_url = mask_obj.get("url")
            if mask_url:
                try:
                    poly_points = get_polygon_from_mask_url(mask_url)
                    svg_data.append({
                        "id": f"obj_{idx}",
                        "points": poly_points # Returns the actual vector points
                    })
                except Exception as e:
                    print(f"Skipping mask {idx} due to error: {e}")

        print(f"✅ Found {len(svg_data)} vector objects.")
        return {"masks": svg_data}

    except Exception as e:
        print(f"❌ Error in API: {str(e)}")
        # We return 200 with an error field to help Figma debug without crashing
        return {"masks": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Railway provides the PORT via environment variables
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)