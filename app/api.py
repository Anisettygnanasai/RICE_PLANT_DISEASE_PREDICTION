from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference import RiceDiseasePredictor
from PIL import Image
import io
import os
from pathlib import Path

app = FastAPI(
    title="Rice Plant Disease API",
    description="API for classifying rice leaf diseases using ResNet18",
    version="1.0.0"
)

# Enable CORS so the React frontend can call it from a different domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your exact domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictor globally so it's ready when requests come in
root_dir = Path(__file__).parent.parent
checkpoint_path = root_dir / "checkpoints" / "best_model.pth"

# Only instantiate if the file exists to prevent startup crashes before training
predictor = None
if checkpoint_path.exists():
    try:
        predictor = RiceDiseasePredictor(checkpoint_path=str(checkpoint_path))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Accepts an image upload and returns disease predictions."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not trained or loaded yet.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    try:
        # Read the uploaded file into a PIL Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run inference
        result = predictor.predict(image)
        return {"status": "success", "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
