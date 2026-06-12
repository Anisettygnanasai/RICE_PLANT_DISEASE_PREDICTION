import argparse
import yaml
import json
import logging
import torch
import wandb
from pathlib import Path
from sklearn.metrics import classification_report

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import get_device, load_class_names, compute_metrics
from src.visualize import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
    return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = get_device(config)
    
    # Load data
    _, _, test_loader, class_names, _ = get_dataloaders(config)
    
    # Build model & load weights
    model = build_model(num_classes=len(class_names), config=config)
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    logging.info("Running evaluation on test set...")
    preds, labels, probs = evaluate(model, test_loader, device)
    
    # Metrics
    metrics = compute_metrics(labels.numpy(), preds.numpy(), probs.numpy(), class_names)
    logging.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    
    # Classification Report
    report = classification_report(labels.numpy(), preds.numpy(), target_names=class_names)
    print("\nClassification Report:\n")
    print(report)
    
    # Save Report
    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    with open(outputs_dir / 'classification_report.txt', 'w') as f:
        f.write("Classification Report:\n\n")
        f.write(report)
        f.write(f"\nOverall Accuracy: {metrics['accuracy']:.4f}\n")
        
    # Confusion Matrix
    cm_path = outputs_dir / 'confusion_matrix.png'
    plot_confusion_matrix(labels.numpy(), preds.numpy(), class_names, str(cm_path))
    logging.info(f"Confusion matrix saved to {cm_path}")
    
    # W&B
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name="test_evaluation",
            job_type="eval"
        )
        
        for k, v in metrics.items():
            wandb.summary[f"test_{k}"] = v
            
        wandb.log({"test_confusion_matrix": wandb.Image(str(cm_path))})
        wandb.finish()
        
    logging.info("Evaluation complete.")

if __name__ == "__main__":
    main()
