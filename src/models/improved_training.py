#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training functions for the improved Pokémon card classifier.
"""

import os
import json
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.models.improved_model import ImprovedPokemonCardClassifier, FocalLoss, save_improved_model
from src.models.improved_utils import load_balanced_data, plot_metrics, plot_confusion_matrix
from src.models.model import DeviceDataLoader, get_default_device


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


def train(model, data_loaders, criterion, optimizer, scheduler, num_epochs, device, save_dir, early_stopping_patience=7):
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
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_results['loss'])
        else:
            scheduler.step()
        
        # Update metrics
        train_metrics['loss'].append(train_results['loss'])
        train_metrics['accuracy'].append(train_results['accuracy'])
        val_metrics['loss'].append(val_results['loss'])
        val_metrics['accuracy'].append(val_results['accuracy'])
        
        # Print metrics
        print(f"Train Loss: {train_results['loss']:.4f}, Train Accuracy: {train_results['accuracy']:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f}, Val Accuracy: {val_results['accuracy']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check if this is the best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            early_stopping_counter = 0
            
            # Save the model
            save_improved_model(model, save_dir, data_loaders['class_names'])
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


def train_improved_model(args):
    """
    Train the improved model.
    
    Args:
        args: Command-line arguments
    """
    # Set the device
    device = get_default_device()
    print(f"Using device: {device}")
    
    # Load the data with class balancing
    print("Loading data...")
    data = load_balanced_data(
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
    model = ImprovedPokemonCardClassifier(
        num_classes=len(data['class_names']),
        model_type=args.model_type,
        pretrained=True
    )
    model.to(device)
    
    # Define the loss function and optimizer
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=data['class_weights'].to(device), gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=data['class_weights'].to(device))
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Define the learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.cosine_t0,
            T_mult=args.cosine_t_mult,
            eta_min=args.min_lr
        )
    else:
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
