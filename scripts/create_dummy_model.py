import torch
import yaml
import json
import os
from pathlib import Path
from src.model import build_model

def create_dummy_model():
    print("Creating dummy model for testing...")
    
    # Ensure directories exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Dummy class names
    classes = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut", "Healthy"]
    with open('class_names.json', 'w') as f:
        json.dump(classes, f)
        
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    config['project']['device'] = 'cpu'
    
    # Build model (with random weights)
    model = build_model(num_classes=len(classes), config=config)
    
    # Save checkpoint
    state = {
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'metrics': {'val_accuracy': 0.99}
    }
    
    torch.save(state, 'checkpoints/best_model.pth')
    print("Saved dummy model to checkpoints/best_model.pth")

if __name__ == "__main__":
    create_dummy_model()
