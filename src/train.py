import argparse
import yaml
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import wandb
import time

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (
    set_seed, get_device, EarlyStopping, CheckpointManager, 
    compute_metrics, AverageMeter, load_class_names
)
from src.visualize import plot_training_curves, log_wandb_predictions_table

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
    return loss_meter.avg, acc_meter.avg

def validate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = [] # for logging samples
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            loss_meter.update(loss.item(), images.size(0))
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
            # Save first batch for visualization
            if i == 0:
                all_images = images.cpu()
                
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Calculate accuracy
    acc = (all_preds == all_labels).float().mean().item()
    
    return loss_meter.avg, acc, all_preds, all_labels, all_probs, all_images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    set_seed(config['project']['seed'])
    device = get_device(config)
    logging.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names, class_weights = get_dataloaders(config)
    
    model = build_model(num_classes=len(class_names), config=config)
    
    # Loss, optimizer, scheduler
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, 
        label_smoothing=config['training']['label_smoothing']
    )
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler_type = config['training']['scheduler']
    epochs = config['training']['epochs']
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    else:
        scheduler = None
        
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta'],
        monitor=config['early_stopping']['monitor'],
        mode='min' if config['early_stopping']['monitor'] == 'val_loss' else 'max'
    )
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config['checkpoint']['dir'],
        metric=config['checkpoint']['metric'],
        mode='max' if config['checkpoint']['metric'] == 'val_accuracy' else 'min'
    )
    
    start_epoch = 1
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = checkpoint_manager.load(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        
    if config['wandb']['enabled']:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name=f"run-{int(time.time())}"
        )
        
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    logging.info("Starting training...")
    for epoch in range(start_epoch, epochs + 1):
        logging.info(f"Epoch {epoch}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels, val_probs, val_batch_imgs = validate(model, val_loader, criterion, device)
        
        if scheduler_type == 'cosine':
            scheduler.step()
        elif scheduler_type == 'plateau':
            scheduler.step(val_loss)
            
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr
        }
        
        logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        if config['wandb']['enabled']:
            wandb.log(metrics)
            
            if config['wandb']['log_images'] and epoch % 5 == 0: # log every 5 epochs
                log_wandb_predictions_table(
                    val_batch_imgs[:16], val_preds[:16], val_labels[:16], val_probs[:16], class_names
                )
                
        checkpoint_manager.save(model, optimizer, scheduler, epoch, metrics)
        
        monitor_val = val_loss if config['early_stopping']['monitor'] == 'val_loss' else val_acc
        if config['early_stopping']['enabled'] and early_stopping(monitor_val):
            logging.info("Early stopping triggered.")
            break
            
    logging.info("Training completed.")
    
    # Save training curves
    plot_training_curves(history)
    
    if config['wandb']['enabled']:
        wandb.finish()
        
    logging.info("Please run `python -m src.evaluate --checkpoint checkpoints/best_model.pth` for final test evaluation.")

if __name__ == "__main__":
    main()
