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

def format_disease_name(raw_name: str) -> str:
    """Enterprise label cleaning: Extracts only the disease name."""
    name = raw_name
    
    # 1. Remove dataset-specific suffixes
    import re
    suffixes = [r"(?i)\s+in\s+rice\s+leaf", r"(?i)\s+in\s+corn\s+leaf", r"(?i)\s+in\s+plant"]
    for s in suffixes:
        name = re.sub(s, "", name)
        
    # 2. Remove plant prefixes
    prefixes = [
        "Apple ", "Blueberry ", "Cherry (including sour) ", "Cherry (including_sour) ",
        "Corn (maize) ", "Grape ", "Peach ", "Pepper bell ", "Potato ", 
        "Raspberry ", "Soybean ", "Strawberry ", "Tomato ", "Garlic ", "Sogatella "
    ]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            
    name = name.strip()
    if name.lower() == "healthy":
        return "Healthy"
    
    # Capitalize first letter cleanly
    return name[0].upper() + name[1:] if name else name

def load_knowledge_base():
    kb_path = Path('app/knowledge_base.json')
    if kb_path.exists():
        with open(kb_path, 'r') as f:
            return json.load(f)
    return {"Diseases": {}, "Categories": {}}

FALLBACK_INFO = {
    "description": "Specific disease profile not found in current database. AI has classified this based on visual anomaly patterns.",
    "causes": "Various environmental, fungal, bacterial, or viral pathogens specific to this plant variety.",
    "prevention": "Isolate affected plants. Consult your local agricultural extension or an expert agronomist for specific treatment."
}

class PlantDiseasePredictor:
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
            
        # Load Enterprise Knowledge Base
        self.knowledge_base = load_knowledge_base()
            
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
            raw_cls_name = self.class_names[top3_indices[i].item()]
            clean_cls_name = format_disease_name(raw_cls_name)
            conf = float(top3_probs[i].item())
            top3_list.append({"class": clean_cls_name, "confidence": conf})
            
        predicted_class = top3_list[0]["class"]
        confidence = top3_list[0]["confidence"]
        
        # AI Reasoning Engine: Look up exact disease, or infer based on category keywords
        raw_pred_class = self.class_names[top3_indices[0].item()]
        
        info = FALLBACK_INFO
        diseases_db = self.knowledge_base.get("Diseases", {})
        categories_db = self.knowledge_base.get("Categories", {})
        
        if predicted_class in diseases_db:
            info = diseases_db[predicted_class]
        elif raw_pred_class in diseases_db:
            info = diseases_db[raw_pred_class]
        else:
            # Fallback Reasoning Engine
            for cat, cat_info in categories_db.items():
                if cat.lower() in predicted_class.lower() or cat.lower() in raw_pred_class.lower():
                    info = {
                        "description": f"Specific disease not profiled, but identified as a type of {cat}. " + cat_info.get("description", ""),
                        "causes": cat_info.get("causes", ""),
                        "prevention": cat_info.get("prevention", "")
                    }
                    break
        
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
        predictor = PlantDiseasePredictor(checkpoint_path=args.checkpoint)
        result = predictor.predict(args.image)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Note: Ensure the model has been trained and 'best_model.pth' and 'class_names.json' exist.")
