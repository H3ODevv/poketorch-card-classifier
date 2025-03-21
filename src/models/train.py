#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the Pokémon card classifier.
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.model import PokemonCardClassifier, save_model, get_default_device, DeviceDataLoader


def get_transforms(img_size=224):
    """
    Get the transforms for training and validation.
    
    Args:
        img_size: Size of the input images
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    # Define the transforms for training
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


def load_data(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Load the dataset.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training and validation
        img_size: Size of the input images
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with 'train_loader', 'val_loader', 'class_names', and 'class_to_idx'
    """
    # Get the transforms
    transforms_dict = get_transforms(img_size)
    
    # Load the datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transforms_dict['train']
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=transforms_dict['val']
    )
    
    # Create the dataloaders
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
    
    # Get the class names and class-to-index mapping
    class_names = [c for c in train_dataset.classes]
    class_to_idx = train_dataset.class_to_idx
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'class_names': class_names,
        'class_to_idx': class_to_idx
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for the training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    # Set the model to training mode
    model.train()
    
    # Initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Train the model
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        # Move the inputs and targets to the device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # Calculate metrics
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy
    }


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for the validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary with 'loss', 'accuracy', 'predictions', and 'targets'
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize lists for predictions and targets
    all_predictions = []
    all_targets = []
    
    # Validate the model
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating", leave=False):
            # Move the inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy,
        'predictions': all_predictions,
        'targets': all_targets
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


def train(model, data_loaders, criterion, optimizer, scheduler, num_epochs, device, save_dir, early_stopping_patience=5):
    """
    Train the model.
    
    Args:
        model: The model to train
        data_loaders: Dictionary with 'train_loader' and 'val_loader'
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on
        save_dir: Directory to save the model and plots
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Dictionary with training and validation metrics
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metrics
    train_metrics = {
        'loss': [],
        'accuracy': []
    }
    
    val_metrics = {
        'loss': [],
        'accuracy': []
    }
    
    # Initialize early stopping variables
    best_val_accuracy = 0.0
    early_stopping_counter = 0
    
    # Train the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_results = train_epoch(
            model=model,
            dataloader=data_loaders['train_loader'],
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        # Validate the model
        val_results = validate(
            model=model,
            dataloader=data_loaders['val_loader'],
            criterion=criterion,
            device=device
        )
        
        # Update the learning rate
        scheduler.step(val_results['loss'])
        
        # Update metrics
        train_metrics['loss'].append(train_results['loss'])
        train_metrics['accuracy'].append(train_results['accuracy'])
        val_metrics['loss'].append(val_results['loss'])
        val_metrics['accuracy'].append(val_results['accuracy'])
        
        # Print metrics
        print(f"Train Loss: {train_results['loss']:.4f}, Train Accuracy: {train_results['accuracy']:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}, Val Accuracy: {val_results['accuracy']:.4f}")
        
        # Check if this is the best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            early_stopping_counter = 0
            
            # Save the model
            save_model(model, save_dir, data_loaders['class_names'])
            print(f"Saved model with validation accuracy: {best_val_accuracy:.4f}")
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter} out of {early_stopping_patience}")
            
            # Check if we should stop early
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping")
                break
    
    # Plot the metrics
    plot_metrics(train_metrics, val_metrics, save_dir)
    
    # Calculate the confusion matrix
    model.eval()
    val_results = validate(
        model=model,
        dataloader=data_loaders['val_loader'],
        criterion=criterion,
        device=device
    )
    
    cm = confusion_matrix(val_results['targets'], val_results['predictions'])
    plot_confusion_matrix(cm, data_loaders['class_names'], save_dir)
    
    # Generate a classification report
    report = classification_report(
        val_results['targets'],
        val_results['predictions'],
        target_names=data_loaders['class_names'],
        output_dict=True
    )
    
    # Save the report
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_val_accuracy': best_val_accuracy
    }


def main(args):
    """
    Main function.
    
    Args:
        args: Command-line arguments
    """
    # Set the device
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Load the data
    print("Loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # Wrap the dataloaders to move data to the device
    data['train_loader'] = DeviceDataLoader(data['train_loader'], device)
    data['val_loader'] = DeviceDataLoader(data['val_loader'], device)
    
    # Print dataset information
    print(f"Number of training samples: {len(data['train_loader'].dataloader.dataset)}")
    print(f"Number of validation samples: {len(data['val_loader'].dataloader.dataset)}")
    print(f"Number of classes: {len(data['class_names'])}")
    
    # Create the model
    print("Creating model...")
    model = PokemonCardClassifier(
        num_classes=len(data['class_names']),
        model_type=args.model_type,
        pretrained=True
    )
    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    results = train(
        model=model,
        data_loaders=data,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir,
        early_stopping_patience=args.patience
    )
    end_time = time.time()
    
    # Print the results
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Pokémon card classifier')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='models/pokemon_card_classifier', help='Directory to save the model')
    parser.add_argument('--model_type', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='Type of ResNet model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--img_size', type=int, default=224, help='Size of the input images')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    main(args)
