# Plant Disease Predictor

## 1. Project Overview & Problem Statement
This project provides a robust, production-ready **Plant Leaf Disease Classification System** using deep transfer learning. Early detection of plant leaf diseases is critical for maximizing agricultural yield and preventing crop loss. This system takes raw leaf images and outputs disease predictions, confidence scores, and basic agronomic recommendations.

## 2. Dataset Description
The model is designed to be trained on standard public agricultural datasets.
- **Expected Classes:** Bacterial Leaf Blight, Brown Spot, Leaf Smut, Healthy (extensible to others like Tungro, Blast, Sheath Blight).
- **Split Strategy:** 70% Train, 15% Validation, 15% Test.

## 3. Setup Instructions
Clone the repository and set up a Python virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 4. Dataset Preparation
Place your raw downloaded dataset in `data/raw/` so that each class has its own subfolder (e.g., `data/raw/BrownSpot/*.jpg`). Then run the dataset preparation script to create a stratified split:
```bash
python scripts/prepare_dataset.py
```
This will populate `data/processed/` with `train`, `val`, and `test` subdirectories.

## 5. Training
To train the model using ResNet18 and Albumentations, run:
```bash
python -m src.train --config configs/config.yaml
```
*Note: Ensure your Weights & Biases (W&B) entity is configured in `configs/config.yaml` if you wish to track the experiment.*

## 6. Evaluation
After training, evaluate the best model checkpoint on the test set:
```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pth
```
This will generate `outputs/classification_report.txt` and `outputs/confusion_matrix.png`.

## 7. Inference

### CLI Usage (Single Image)
```bash
python -m src.inference --image samples/test_leaf.jpg --checkpoint checkpoints/best_model.pth
```

### Python API Example
```python
from src.inference import PlantDiseasePredictor

# Initialize predictor
predictor = PlantDiseasePredictor(checkpoint_path="checkpoints/best_model.pth")

# Predict
result = predictor.predict("path/to/leaf.jpg")
print(result)
```

## 8. Results
*(To be filled after final model training)*

| Metric | Train | Validation | Test |
| :--- | :--- | :--- | :--- |
| **Accuracy** | TBD | TBD | TBD |
| **Loss** | TBD | TBD | TBD |

## 9. Project Structure Tree
```text
plant-disease-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── configs/
│   └── config.yaml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── utils.py
│   └── visualize.py
├── scripts/
│   ├── prepare_dataset.py
│   └── mock_data.py
├── checkpoints/
├── outputs/
├── app/
├── requirements.txt
└── .gitignore
```

## 10. W&B Dashboard
[Weights & Biases Dashboard Link] *(Placeholder - update with actual run link)*

## 11. Limitations & Disclaimer
> **Disclaimer:** This model provides general guidance based on visual leaf symptoms. It is not a substitute for expert agronomic diagnosis. Always consult a local agricultural extension officer for confirmed treatment plans.
