import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_dataset(raw_dir, processed_dir, seed=42):
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    if not raw_path.exists():
        logging.error(f"Raw data directory {raw_path} does not exist.")
        logging.info("Please place your dataset in data/raw/ such that each class has its own subfolder.")
        return

    # Gather all images
    classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    if not classes:
        logging.error(f"No class directories found in {raw_path}")
        return

    all_images = []
    all_labels = []
    
    for cls in classes:
        cls_path = raw_path / cls
        images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpeg"))
        all_images.extend(images)
        all_labels.extend([cls] * len(images))
        
    if not all_images:
        logging.error("No images found in the class directories.")
        return
        
    logging.info(f"Found {len(all_images)} images across {len(classes)} classes.")

    # Stratified split: 70% train, 30% temp (which will be 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.30, random_state=seed, stratify=all_labels
    )
    
    # Split temp into val and test (50% of 30% = 15% each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    
    logging.info(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Function to copy files
    def copy_files(files, labels, split_name):
        for f, label in zip(files, labels):
            dest_dir = processed_path / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest_dir / f.name)
            
    # Clear processed dir if exists
    if processed_path.exists():
        logging.info(f"Clearing existing processed directory: {processed_path}")
        shutil.rmtree(processed_path)
        
    logging.info("Copying files to processed directory...")
    copy_files(X_train, y_train, 'train')
    copy_files(X_val, y_val, 'val')
    copy_files(X_test, y_test, 'test')
    logging.info("Dataset preparation complete!")

if __name__ == "__main__":
    config = load_config()
    prepare_dataset(
        raw_dir="data/raw",
        processed_dir=config['data']['data_dir'],
        seed=config['project']['seed']
    )
