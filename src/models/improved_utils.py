#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the improved Pokémon card classifier.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from src.models.model import DeviceDataLoader, get_default_device


def get_enhanced_transforms(img_size=224):
    """
    Get enhanced transforms for training and validation.
    
    Args:
        img_size: Size of the input images
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    # Define the transforms for training with enhanced augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),  # Increased rotation range
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Add affine transformations
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Enhanced color jitter
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective changes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define the transforms for validation
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transforms,
        'val': val_transforms
    }


def load_balanced_data(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Load the dataset with class balancing.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training and validation
        img_size: Size of the input images
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with 'train_loader', 'val_loader', 'class_names', and 'class_to_idx'
    """
    # Get the transforms
    transforms_dict = get_enhanced_transforms(img_size)
    
    # Load the datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transforms_dict['train']
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=transforms_dict['val']
    )
    
    # Calculate class weights for balanced sampling
    class_counts = Counter(train_dataset.targets)
    class_weights = {class_idx: 1.0 / count for class_idx, count in class_counts.items()}
    sample_weights = [class_weights[target] for target in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create the dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
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
    
    # Get the class names and class-to-index mapping
    class_names = [c for c in train_dataset.classes]
    class_to_idx = train_dataset.class_to_idx
    
    # Calculate class weights for loss function
    class_weights_tensor = torch.FloatTensor([1.0 / class_counts[i] for i in range(len(class_names))])
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum() * len(class_names)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'class_weights': class_weights_tensor
    }


def plot_metrics(train_metrics, val_metrics, save_dir):
    """
    Plot the training and validation metrics.
    
    Args:
        train_metrics: Dictionary with training metrics
        val_metrics: Dictionary with validation metrics
        save_dir: Directory to save the plots
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    
    # Plot the accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics['accuracy'], label='Training Accuracy')
    plt.plot(val_metrics['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()


def plot_confusion_matrix(cm, class_names, save_dir):
    """
    Plot the confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_dir: Directory to save the plot
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot the confusion matrix
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add the values to the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
