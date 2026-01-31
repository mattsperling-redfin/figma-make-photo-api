import os
import cv2
import numpy as np
import requests
import fal_client
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutoSegmentRequest(BaseModel):
    image_url: str

def get_polygon_from_mask_url(url):
    # Download the PNG mask
    resp = requests.get(url)
    nparr = np.frombuffer(resp.content, np.uint8)
    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Find contours (the outline of the mask)
    contours, _ = cv2.find_contours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        # Simplify the polygon to keep the JSON small
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Convert to a flat list of [x, y, x, y...]
        points = approx.reshape(-1, 2).tolist()
        polygons.append(points)
    return polygons

@app.post("/segment")
async def segment_to_svg(request: AutoSegmentRequest):
    print(f"Segmenting: {request.image_url}")
    
    # 1. Get PNG masks from SAM 2
    result = fal_client.subscribe(
        "fal-ai/sam2/auto-segment",
        arguments={"image_url": request.image_url}
    )
    
    # 2. Convert PNGs to Polygons
    individual_masks = result.get("individual_masks", [])
    svg_data = []
    
    for idx, mask_obj in enumerate(individual_masks):
        mask_url = mask_obj.get("url")
        try:
            poly_points = get_polygon_from_mask_url(mask_url)
            svg_data.append({
                "id": f"obj_{idx}",
                "points": poly_points
            })
        except Exception as e:
            print(f"Error processing mask {idx}: {e}")

    return {"masks": svg_data}