import json
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Union, List, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import argparse

from src.model import build_model
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.utils import get_device

# Structured Enterprise Intelligence Dictionary
DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "description": "A devastating bacterial disease caused by Xanthomonas oryzae, leading to wilting and death of seedlings.",
        "causes": "Spreads through wind, rain, and irrigation water, especially in high humidity and temperatures (25-34°C).",
        "prevention": "Apply copper-based bactericides; use resistant seed varieties; avoid excess nitrogen fertilizers; ensure proper field drainage."
    },
    "Brown Spot": {
        "description": "A fungal disease caused by Bipolaris oryzae, manifesting as brown, oval spots on leaves.",
        "causes": "Common in soils with poor fertility, nutrient deficiency (especially potassium), and periods of water stress.",
        "prevention": "Improve soil fertility with balanced NPK fertilizers; apply recommended fungicides (e.g., Propiconazole); avoid water stress."
    },
    "Leaf Smut": {
        "description": "A fungal disease caused by Entyloma oryzae, creating small, black, linear spots (smuts) on leaves.",
        "causes": "Favored by high humidity, frequent rainfall, and excessive nitrogen application.",
        "prevention": "Use certified disease-free seeds; practice crop rotation; apply appropriate fungicides during the early tillering stage."
    },
    "Healthy": {
        "description": "The plant exhibits normal growth patterns with no visual symptoms of disease or nutrient deficiency.",
        "causes": "Optimal environmental conditions, good soil health, and proper agricultural practices.",
        "prevention": "Continue regular monitoring, maintain balanced fertilization, and ensure adequate water management."
    },
    "Tungro": {
        "description": "A viral disease causing stunting, yellowing of leaves, and delayed flowering.",
        "causes": "Transmitted primarily by the green leafhopper (Nephotettix virescens).",
        "prevention": "Control leafhopper populations with insecticides; uproot and destroy infected plants immediately; plant resistant varieties."
    },
    "Blast": {
        "description": "Caused by the fungus Magnaporthe oryzae, it produces diamond-shaped lesions on leaves and can infect collars and panicles.",
        "causes": "High humidity, prolonged leaf wetness, and cool nights followed by warm days.",
        "prevention": "Apply systemic fungicides like Tricyclazole; avoid excessive nitrogen; manage field water to avoid drought stress."
    },
    "Sheath Blight": {
        "description": "A major fungal disease caused by Rhizoctonia solani, leading to lesions on the leaf sheath that can spread to the blades.",
        "causes": "High planting density, excessive nitrogen, and high humidity/temperature within the crop canopy.",
        "prevention": "Reduce planting density; clear field of weeds and crop residues; apply appropriate fungicides (e.g., Validamycin)."
    }
}

FALLBACK_INFO = {
    "description": "Specific disease profile not found in current database. AI has classified this based on visual anomaly patterns.",
    "causes": "Various environmental, fungal, bacterial, or viral pathogens specific to this plant variety.",
    "prevention": "Isolate affected plants. Consult your local agricultural extension or an expert agronomist for specific treatment."
}

class RiceDiseasePredictor:
    def __init__(self, checkpoint_path, class_names_path='class_names.json', config_path='configs/config.yaml', device='auto'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Device
        if device != 'auto':
            self.config['project']['device'] = device
        self.device = get_device(self.config)
        
        # Load class names
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
            
        # Build model architecture
        self.model = build_model(num_classes=len(self.class_names), config=self.config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
        self.model.eval()
        self.model.to(self.device)
        
        # Build transform
        img_size = self.config['data']['img_size']
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

    def _preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image, str):
            # Read from path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not read image at {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # Convert PIL to numpy
            img = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            # Assume RGB numpy array
            img = image
        else:
            raise TypeError("Image must be a path string, PIL Image, or numpy array")
            
        augmented = self.transform(image=img)
        tensor = augmented['image'].unsqueeze(0) # Add batch dim
        return tensor.to(self.device)

    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Predicts disease from a single image.
        """
        tensor = self._preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        # Get Top-3
        top3_probs, top3_indices = torch.topk(probs, k=min(3, len(self.class_names)))
        
        top3_list = []
        for i in range(len(top3_indices)):
            cls_name = self.class_names[top3_indices[i].item()]
            conf = float(top3_probs[i].item())
            top3_list.append({"class": cls_name, "confidence": conf})
            
        predicted_class = top3_list[0]["class"]
        confidence = top3_list[0]["confidence"]
        
        info = DISEASE_INFO.get(predicted_class, FALLBACK_INFO)
        
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top3": top3_list,
            "details": info
        }
        
        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predicts over multiple image paths sequentially.
        """
        results = []
        for path in image_paths:
            try:
                res = self.predict(path)
                res["image_path"] = path
                results.append(res)
            except Exception as e:
                results.append({"image_path": path, "error": str(e)})
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the leaf image.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint.")
    
    args = parser.parse_args()
    
    try:
        predictor = RiceDiseasePredictor(checkpoint_path=args.checkpoint)
        result = predictor.predict(args.image)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Note: Ensure the model has been trained and 'best_model.pth' and 'class_names.json' exist.")
