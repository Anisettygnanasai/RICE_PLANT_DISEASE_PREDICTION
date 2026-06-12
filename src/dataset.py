import os
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class RiceLeafDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.split = split
        
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Read image with OpenCV (BGR) and convert to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def get_transforms(config):
    img_size = config['data']['img_size']
    aug_train = config['augmentation']['train']
    
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=aug_train['horizontal_flip']),
        A.VerticalFlip(p=aug_train['vertical_flip']),
        A.Rotate(limit=aug_train['rotate_limit'], p=0.5),
        A.RandomBrightnessContrast(p=aug_train['brightness_contrast']),
        A.HueSaturationValue(p=aug_train['hue_saturation']),
        A.GaussianBlur(p=aug_train['gaussian_blur']),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=aug_train['coarse_dropout']),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    val_test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    
    return train_transform, val_test_transform

def save_class_names(classes, path):
    with open(path, 'w') as f:
        json.dump(classes, f, indent=4)

def get_class_weights(dataset):
    class_counts = Counter(dataset.labels)
    total_samples = len(dataset)
    num_classes = len(class_counts)
    
    weights = []
    for i in range(num_classes):
        # Inverse class frequency formulation
        weight = total_samples / (num_classes * class_counts[i])
        weights.append(weight)
        
    return torch.FloatTensor(weights)

def get_dataloaders(config):
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    class_names_path = config['data']['class_names_path']
    
    train_transform, val_test_transform = get_transforms(config)
    
    train_dataset = RiceLeafDataset(data_dir, split='train', transform=train_transform)
    val_dataset = RiceLeafDataset(data_dir, split='val', transform=val_test_transform)
    test_dataset = RiceLeafDataset(data_dir, split='test', transform=val_test_transform)
    
    # Save class names mapping
    save_class_names(train_dataset.classes, class_names_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    class_weights = get_class_weights(train_dataset)
    
    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights

if __name__ == "__main__":
    import yaml
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # This acts as a quick test if run directly.
    # Assumes dummy data is generated first for testing, otherwise will fail.
    print("Testing get_dataloaders...")
    try:
        train_loader, val_loader, test_loader, classes, weights = get_dataloaders(config)
        print(f"Classes: {classes}")
        print(f"Class Weights: {weights}")
        for imgs, labels in train_loader:
            print(f"Train Batch Image Shape: {imgs.shape}")
            print(f"Train Batch Label Shape: {labels.shape}")
            break
        print("Dataloader test passed.")
    except Exception as e:
        print(f"Dataloader test could not complete (probably no data yet): {e}")
