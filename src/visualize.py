import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix
from pathlib import Path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denormalize_image(tensor):
    """Denormalize a tensor image back to numpy array for plotting."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img

def plot_training_curves(history, save_path="outputs/training_curves.png"):
    """Plots training and validation loss and accuracy curves."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    if 'val_acc' in history and len(history['val_acc']) > 0:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="outputs/confusion_matrix.png"):
    """Plots a confusion matrix using seaborn."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_sample_predictions(model, dataset, class_names, device, n=9, save_path="outputs/sample_predictions.png"):
    """Plots a grid of sample predictions."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = dataset[idx]
            # Predict
            img_tensor = image.unsqueeze(0).to(device)
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            pred_idx = preds.item()
            
            true_name = class_names[label]
            pred_name = class_names[pred_idx]
            
            # Plot
            ax = axes[i]
            img_disp = denormalize_image(image)
            ax.imshow(img_disp)
            
            color = 'green' if label == pred_idx else 'red'
            ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color)
            ax.axis('off')
            
    # Hide any unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def log_wandb_predictions_table(images, preds, labels, probs, class_names):
    """Logs a table of sample predictions to Weights & Biases."""
    table = wandb.Table(columns=["Image", "True Label", "Predicted Label", "Confidence"])
    
    for i in range(len(images)):
        img = denormalize_image(images[i])
        
        # Convert to uint8 for wandb.Image
        img = (img * 255).astype(np.uint8)
        wandb_img = wandb.Image(img)
        
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        confidence = probs[i].max().item()
        
        table.add_data(wandb_img, true_label, pred_label, f"{confidence:.4f}")
        
    wandb.log({"val_predictions": table})
