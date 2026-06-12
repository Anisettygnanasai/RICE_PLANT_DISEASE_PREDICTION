import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import logging

def build_model(num_classes, config):
    model_config = config['model']
    
    # 1. Load pre-trained ResNet18
    if model_config['pretrained']:
        weights = ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None
        
    model = resnet18(weights=weights)
    
    # 2. Freeze backbone if requested
    if model_config['freeze_backbone']:
        for param in model.parameters():
            param.requires_grad = False
            
        # Optional: unfreeze layer4 for better fine-tuning
        # for param in model.layer4.parameters():
        #     param.requires_grad = True
            
    # 3. Replace the final fc layer
    dropout_p = model_config['dropout']
    
    # ResNet18 features before fc are 512-dimensional
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=dropout_p * 0.5),
        nn.Linear(in_features=256, out_features=num_classes)
    )
    
    device_name = config['project']['device']
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
        
    model = model.to(device)
    
    # Print model summary
    total_params, trainable_params = count_parameters(model)
    logging.info(f"Model built. Total params: {total_params:,}. Trainable params: {trainable_params:,}")
    
    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Testing build_model...")
    # Assume 4 classes for test
    model = build_model(num_classes=4, config=config)
    print(model.fc)
