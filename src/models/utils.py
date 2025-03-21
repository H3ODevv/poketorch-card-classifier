#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Pokémon card classifier.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from src.models.model import load_model


def evaluate_model(model, dataloader, device=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: Device to evaluate on (None for auto-detection)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize metrics
    correct = 0
    total = 0
    
    # Initialize lists for predictions and targets
    all_predictions = []
    all_targets = []
    
    # Evaluate the model
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move the inputs and targets to the device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update metrics
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    
    # Calculate the confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Generate a classification report
    report = classification_report(
        all_targets,
        all_predictions,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'targets': all_targets
    }


def visualize_predictions(model, dataloader, class_names, num_samples=5, device=None):
    """
    Visualize model predictions on a few samples.
    
    Args:
        model: The model to use
        dataloader: DataLoader for the dataset
        class_names: List of class names
        num_samples: Number of samples to visualize
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        None (displays the visualizations)
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the model to evaluation mode
    model.eval()
    
    # Get a batch of data
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Move the images and labels to the device
    images = images.to(device)
    labels = labels.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
    
    # Convert images for display
    images = images.cpu().numpy()
    
    # Plot the images
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        # Get the image
        img = np.transpose(images[i], (1, 2, 0))
        
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot the image
        axes[i].imshow(img)
        axes[i].set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[predicted[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def predict_image_from_path(image_path, model_dir, device=None):
    """
    Predict the class of an image from a file path.
    
    Args:
        image_path: Path to the image
        model_dir: Directory containing the model
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        Dictionary with prediction results
    """
    # Load the model
    model, class_names = load_model(model_dir, device)
    
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Make the prediction
    return model.predict(img, class_names, device)


def predict_image_from_array(image_array, model_dir, device=None):
    """
    Predict the class of an image from a numpy array.
    
    Args:
        image_array: Numpy array containing the image (H, W, C)
        model_dir: Directory containing the model
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        Dictionary with prediction results
    """
    # Load the model
    model, class_names = load_model(model_dir, device)
    
    # Convert the array to a PIL image
    img = Image.fromarray(image_array.astype('uint8')).convert('RGB')
    
    # Make the prediction
    return model.predict(img, class_names, device)


def batch_predict(image_paths, model_dir, device=None):
    """
    Predict the classes of multiple images.
    
    Args:
        image_paths: List of paths to the images
        model_dir: Directory containing the model
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        List of dictionaries with prediction results
    """
    # Load the model
    model, class_names = load_model(model_dir, device)
    
    # Initialize the results list
    results = []
    
    # Process each image
    for path in image_paths:
        # Load the image
        img = Image.open(path).convert('RGB')
        
        # Make the prediction
        prediction = model.predict(img, class_names, device)
        
        # Add the path to the prediction
        prediction['image_path'] = path
        
        # Add the prediction to the results
        results.append(prediction)
    
    return results


def export_model_info(model_dir, output_path):
    """
    Export model information to a JSON file.
    
    Args:
        model_dir: Directory containing the model
        output_path: Path to save the model information
        
    Returns:
        None
    """
    # Load the class names
    class_names_path = os.path.join(model_dir, 'class_names.json')
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    # Load the classification report if it exists
    report_path = os.path.join(model_dir, 'classification_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
    else:
        report = None
    
    # Create the model information
    model_info = {
        'num_classes': len(class_names),
        'class_names': class_names,
        'classification_report': report
    }
    
    # Save the model information
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=4)


def create_model_summary(model_dir, output_path):
    """
    Create a summary of the model.
    
    Args:
        model_dir: Directory containing the model
        output_path: Path to save the model summary
        
    Returns:
        None
    """
    # Load the class names
    class_names_path = os.path.join(model_dir, 'class_names.json')
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    # Load the classification report if it exists
    report_path = os.path.join(model_dir, 'classification_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
    else:
        report = None
    
    # Create the summary
    summary = []
    summary.append("# Pokémon Card Classifier Model Summary")
    summary.append("")
    summary.append(f"Number of classes: {len(class_names)}")
    summary.append("")
    
    # Add the classification report
    if report is not None:
        summary.append("## Classification Report")
        summary.append("")
        summary.append("| Class | Precision | Recall | F1-Score | Support |")
        summary.append("|-------|-----------|--------|----------|---------|")
        
        # Add the metrics for each class
        for class_name in class_names:
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1_score = report[class_name]['f1-score']
                support = report[class_name]['support']
                
                summary.append(f"| {class_name} | {precision:.4f} | {recall:.4f} | {f1_score:.4f} | {support} |")
        
        # Add the average metrics
        summary.append(f"| **Accuracy** | | | {report['accuracy']:.4f} | {report['macro avg']['support']} |")
        summary.append(f"| **Macro Avg** | {report['macro avg']['precision']:.4f} | {report['macro avg']['recall']:.4f} | {report['macro avg']['f1-score']:.4f} | {report['macro avg']['support']} |")
        summary.append(f"| **Weighted Avg** | {report['weighted avg']['precision']:.4f} | {report['weighted avg']['recall']:.4f} | {report['weighted avg']['f1-score']:.4f} | {report['weighted avg']['support']} |")
    
    # Save the summary
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary))
