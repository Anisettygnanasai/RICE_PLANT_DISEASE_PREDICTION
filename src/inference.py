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

# Static recommendation dictionary
DISEASE_RECOMMENDATIONS = {
    "Bacterial Leaf Blight": "Apply copper-based bactericide; avoid excess nitrogen; ensure field drainage.",
    "Brown Spot": "Improve soil potassium levels; apply recommended fungicide; avoid water stress.",
    "Leaf Smut": "Use resistant varieties next season; apply fungicide at early tillering stage.",
    "Healthy": "No disease detected. Continue regular monitoring and balanced fertilization.",
    "Tungro": "Control green leafhopper vectors with insecticides; uproot and destroy infected plants.",
    "Blast": "Apply tricyclazole or similar fungicide; avoid excess nitrogen fertilizers; manage field water levels.",
    "Sheath Blight": "Reduce planting density; apply appropriate fungicides; clear field of weeds and crop residues."
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
        
        recommendation = DISEASE_RECOMMENDATIONS.get(
            predicted_class, 
            "General advice: Consult a local agricultural expert for treatment options."
        )
        
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top3": top3_list,
            "recommendation": f"{recommendation} (Note: This is general guidance, not a substitute for expert diagnosis.)"
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
