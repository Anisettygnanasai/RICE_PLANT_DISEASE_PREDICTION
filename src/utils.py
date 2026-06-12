import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import logging
import json

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(config):
    """Auto-detects device based on config."""
    device_name = config['project']['device']
    if device_name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_name)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        if mode == 'min':
            self.val_is_better = lambda a, best: a < best - self.min_delta
        else:
            self.val_is_better = lambda a, best: a > best + self.min_delta

    def __call__(self, current_value) -> bool:
        if self.best_value is None:
            self.best_value = current_value
        elif self.val_is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

class CheckpointManager:
    def __init__(self, checkpoint_dir, metric='val_accuracy', mode='max'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.best_metric_value = None

    def save(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last_checkpoint.pth'
        torch.save(state, last_path)
        
        # Check if best and save
        current_metric = metrics.get(self.metric)
        if current_metric is not None:
            if self.best_metric_value is None:
                is_best = True
                self.best_metric_value = current_metric
            else:
                if self.mode == 'max' and current_metric > self.best_metric_value:
                    is_best = True
                    self.best_metric_value = current_metric
                elif self.mode == 'min' and current_metric < self.best_metric_value:
                    is_best = True
                    self.best_metric_value = current_metric
                    
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(state, best_path)
                logging.info(f"Saved new best model with {self.metric}: {current_metric:.4f}")

    def load(self, path, model, optimizer=None, scheduler=None):
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'], checkpoint['metrics']

def compute_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """Computes evaluation metrics using sklearn."""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    metrics['macro_precision'] = precision
    metrics['macro_recall'] = recall
    metrics['macro_f1'] = f1
    
    # Per-class metrics
    if class_names:
        precision_pc, recall_pc, f1_pc, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
        )
        for i, cls in enumerate(class_names):
            metrics[f'{cls}_precision'] = precision_pc[i]
            metrics[f'{cls}_recall'] = recall_pc[i]
            metrics[f'{cls}_f1'] = f1_pc[i]
            
    return metrics

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_class_names(path):
    with open(path, 'r') as f:
        return json.load(f)
