#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved model architecture for the Pokémon card classifier.
"""

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import os
import json

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weight for each class (tensor of shape [num_classes])
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedPokemonCardClassifier(nn.Module):
    """
    Improved Pokémon card classifier with more model options and enhanced classifier head.
    """
    def __init__(self, num_classes, model_type='efficientnet_b0', pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of Pokémon classes
            model_type: Type of model to use ('efficientnet_b0', 'efficientnet_b3', 'resnet50', etc.)
            pretrained: Whether to use pretrained weights
        """
        super(ImprovedPokemonCardClassifier, self).__init__()
        
        # Select the base model
        if model_type.startswith('efficientnet'):
            # EfficientNet models
            if model_type == 'efficientnet_b0':
                self.base_model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
                num_features = self.base_model.classifier[1].in_features
                self.base_model.classifier = nn.Identity()
            elif model_type == 'efficientnet_b3':
                self.base_model = models.efficientnet_b3(weights='DEFAULT' if pretrained else None)
                num_features = self.base_model.classifier[1].in_features
                self.base_model.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported EfficientNet model: {model_type}")
        elif model_type.startswith('resnet'):
            # ResNet models
            if model_type == 'resnet18':
                self.base_model = models.resnet18(weights='DEFAULT' if pretrained else None)
            elif model_type == 'resnet34':
                self.base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
            elif model_type == 'resnet50':
                self.base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
            elif model_type == 'resnet101':
                self.base_model = models.resnet101(weights='DEFAULT' if pretrained else None)
            else:
                raise ValueError(f"Unsupported ResNet model: {model_type}")
            
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_type == 'convnext_tiny':
            # ConvNeXt models
            self.base_model = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
            num_features = self.base_model.classifier[2].in_features
            self.base_model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Improved classifier head with batch normalization and more layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Define the image transforms for inference
        self.inference_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Store the model type
        self.model_type = model_type
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        features = self.base_model(x)
        return self.classifier(features)
    
    def predict(self, image, class_names, device=None):
        """
        Predict the class of an image.
        
        Args:
            image: PIL Image
            class_names: List of class names
            device: Device to run the prediction on
            
        Returns:
            Dictionary with prediction results
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Preprocess the image
        image_tensor = self.inference_transforms(image).unsqueeze(0).to(device)
        
        # Set the model to evaluation mode
        self.eval()
        
        # Make the prediction
        with torch.no_grad():
            outputs = self(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get the top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Convert to lists
            top5_prob = top5_prob.cpu().numpy().tolist()
            top5_indices = top5_indices.cpu().numpy().tolist()
            top5_classes = [class_names[idx] for idx in top5_indices]
            
            # Get the predicted class
            predicted_idx = top5_indices[0]
            predicted_class = class_names[predicted_idx]
            confidence = top5_prob[0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top5_classes': top5_classes,
            'top5_probabilities': top5_prob
        }


def save_improved_model(model, save_dir, class_names=None):
    """
    Save the improved model and class names.
    
    Args:
        model: The model to save
        save_dir: Directory to save the model
        class_names: List of class names
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(save_dir, 'pokemon_card_classifier.pth')
    torch.save({
        'model_type': model.model_type,
        'num_classes': model.classifier[-1].out_features,
        'state_dict': model.state_dict()
    }, model_path)
    
    # Save the class names
    if class_names is not None:
        class_names_path = os.path.join(save_dir, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)


def load_improved_model(model_dir):
    """
    Load the improved model and class names.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        Tuple of (model, class_names)
    """
    # Load the model
    model_path = os.path.join(model_dir, 'pokemon_card_classifier.pth')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create the model
    model = ImprovedPokemonCardClassifier(
        num_classes=checkpoint['num_classes'],
        model_type=checkpoint['model_type'],
        pretrained=False
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load the class names
    class_names_path = os.path.join(model_dir, 'class_names.json')
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    return model, class_names
