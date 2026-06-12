import os
import cv2
import numpy as np
from pathlib import Path

def create_mock_data(base_dir="data/raw", classes=['BacterialLeafBlight', 'BrownSpot', 'Healthy', 'LeafSmut'], samples_per_class=20):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for cls in classes:
        cls_path = base_path / cls
        cls_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            # Create a random image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(cls_path / f"mock_{i}.jpg"), img)
            
    print(f"Created mock data in {base_dir}")

if __name__ == "__main__":
    create_mock_data()
