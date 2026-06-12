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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root_ui():
    """Serves a simple HTML UI for testing the predictor."""
    status_color = "green" if predictor is not None else "red"
    status_text = "Model Loaded & Ready" if predictor is not None else "Model Not Loaded (Train model first!)"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rice Disease Predictor</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0fdf4; color: #166534; max-width: 800px; margin: 0 auto; padding: 2rem; }}
            .card {{ background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
            h1 {{ color: #15803d; }}
            .status {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; font-weight: bold; background-color: {status_color}; color: white; margin-bottom: 1rem; }}
            .upload-btn {{ background-color: #22c55e; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 6px; cursor: pointer; font-size: 1rem; font-weight: bold; transition: background-color 0.2s; }}
            .upload-btn:hover {{ background-color: #16a34a; }}
            #result {{ margin-top: 2rem; display: none; padding: 1rem; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; color: #334155; }}
            .api-info {{ margin-top: 3rem; font-size: 0.875rem; color: #64748b; border-top: 1px solid #cbd5e1; padding-top: 1rem; }}
            img#preview {{ max-width: 100%; max-height: 300px; margin-top: 1rem; border-radius: 8px; display: none; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🌾 Rice Plant Disease Predictor</h1>
            <div class="status">{status_text}</div>
            <p>Upload a photo of a rice leaf to detect potential diseases.</p>
            
            <form id="uploadForm">
                <input type="file" id="imageInput" accept="image/*" style="margin-bottom: 1rem; display: block;" required>
                <img id="preview" src="" alt="Image preview">
                <br>
                <button type="submit" class="upload-btn">Analyze Leaf</button>
            </form>

            <div id="result">
                <h3 style="margin-top:0;">Prediction Result:</h3>
                <p><strong>Disease:</strong> <span id="resDisease"></span></p>
                <p><strong>Confidence:</strong> <span id="resConfidence"></span>%</p>
                <p><strong>Recommendation:</strong> <br> <span id="resRec"></span></p>
            </div>
            
            <div class="api-info">
                <strong>Developers:</strong> To use this as an API in your React application, send a POST request with the image file to <code>/predict</code>.
            </div>
        </div>

        <script>
            const imageInput = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const form = document.getElementById('uploadForm');
            const resultDiv = document.getElementById('result');

            imageInput.addEventListener('change', function() {{
                const file = this.files[0];
                if (file) {{
                    const reader = new FileReader();
                    reader.onload = function(e) {{
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }}
                    reader.readAsDataURL(file);
                }}
            }});

            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                const file = imageInput.files[0];
                if (!file) return;

                const submitBtn = form.querySelector('button');
                submitBtn.textContent = 'Analyzing...';
                submitBtn.disabled = true;

                const formData = new FormData();
                formData.append('file', file);

                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        document.getElementById('resDisease').textContent = data.data.predicted_class;
                        document.getElementById('resConfidence').textContent = (data.data.confidence * 100).toFixed(2);
                        document.getElementById('resRec').textContent = data.data.recommendation;
                        resultDiv.style.display = 'block';
                    }} else {{
                        alert('Error: ' + (data.detail || 'Failed to analyze image'));
                    }}
                }} catch (error) {{
                    alert('Network error occurred.');
                }} finally {{
                    submitBtn.textContent = 'Analyze Leaf';
                    submitBtn.disabled = false;
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html_content

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
