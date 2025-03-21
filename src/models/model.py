#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model architecture for the Pokémon card classifier.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

class PokemonCardClassifier(nn.Module):
    """
    Pokémon card classifier using transfer learning with ResNet.
    """
    def __init__(self, num_classes, model_type='resnet50', pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of Pokémon classes
            model_type: Type of ResNet model to use ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            pretrained: Whether to use pretrained weights
        """
        super(PokemonCardClassifier, self).__init__()
        
        # Select the base model
        if model_type == 'resnet18':
            self.base_model = models.resnet18(weights='DEFAULT' if pretrained else None)
        elif model_type == 'resnet34':
            self.base_model = models.resnet34(weights='DEFAULT' if pretrained else None)
        elif model_type == 'resnet50':
            self.base_model = models.resnet50(weights='DEFAULT' if pretrained else None)
        elif model_type == 'resnet101':
            self.base_model = models.resnet101(weights='DEFAULT' if pretrained else None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Get the number of features in the last layer
        num_features = self.base_model.fc.in_features
        
        # Replace the final fully connected layer
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
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
        return self.base_model(x)
    
    def predict(self, img, class_names, device=None):
        """
        Predict the class of an image.
        
        Args:
            img: PIL Image
            class_names: List of class names
            device: Device to run inference on (None for auto-detection)
            
        Returns:
            Dictionary with prediction results
        """
        # Auto-detect device if not specified
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare the image
        img = self.inference_transforms(img).unsqueeze(0).to(device)
        
        # Set the model to evaluation mode
        self.eval()
        
        # Make the prediction
        with torch.no_grad():
            outputs = self(img)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # Get the top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5)
            
            # Convert to lists
            top5_probs = top5_probs.cpu().numpy().tolist()
            top5_indices = top5_indices.cpu().numpy().tolist()
            
            # Get the class names
            top5_classes = [class_names[idx] for idx in top5_indices]
            
            # Create the result dictionary
            result = {
                'top5_classes': top5_classes,
                'top5_probabilities': top5_probs,
                'predicted_class': top5_classes[0],
                'confidence': top5_probs[0]
            }
            
            return result


def save_model(model, save_dir, class_names=None):
    """
    Save the model and class names.
    
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
        'num_classes': model.base_model.fc[-1].out_features,
        'state_dict': model.state_dict()
    }, model_path)
    
    # Save the class names
    if class_names is not None:
        class_names_path = os.path.join(save_dir, 'class_names.json')
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f)


def load_model(save_dir, device=None):
    """
    Load the model and class names.
    
    Args:
        save_dir: Directory where the model is saved
        device: Device to load the model on (None for auto-detection)
        
    Returns:
        Tuple of (model, class_names)
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model_path = os.path.join(save_dir, 'pokemon_card_classifier.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create the model
    model = PokemonCardClassifier(
        num_classes=checkpoint['num_classes'],
        model_type=checkpoint['model_type'],
        pretrained=False
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    # Load the class names
    class_names_path = os.path.join(save_dir, 'class_names.json')
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    return model, class_names


def predict_image(image_path, model, class_names, device=None):
    """
    Predict the class of an image.
    
    Args:
        image_path: Path to the image
        model: The model to use
        class_names: List of class names
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        Dictionary with prediction results
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Make the prediction
    return model.predict(img, class_names, device)


def get_default_device():
    """
    Get the default device for training and inference.
    
    Returns:
        torch.device: The default device (CUDA if available, otherwise CPU)
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(data, device):
    """
    Move tensor(s) to the specified device.
    
    Args:
        data: Tensor or collection of tensors
        device: Device to move the tensor(s) to
        
    Returns:
        Tensor(s) moved to the device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """
    Wrap a dataloader to move data to a device.
    """
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
    
    def __iter__(self):
        for batch in self.dataloader:
            yield to_device(batch, self.device)
    
    def __len__(self):
        return len(self.dataloader)
