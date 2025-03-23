#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the Pokémon card classifier API.
"""

import os
import json
import torch
from PIL import Image
from functools import lru_cache

from src.models.model import load_model
from src.models.improved_model import load_improved_model

# Global variables to store the model and class names
_model = None
_class_names = None

@lru_cache(maxsize=1)
def load_model_singleton(model_dir):
    """
    Load the model as a singleton to avoid loading it multiple times.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        Tuple of (model, class_names)
    """
    global _model, _class_names
    
    if _model is None or _class_names is None:
        try:
            # Check if this is an improved model
            model_path = os.path.join(model_dir, 'pokemon_card_classifier.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_type' in checkpoint:
                    # This is an improved model
                    _model, _class_names = load_improved_model(model_dir)
                else:
                    # This is the original model
                    _model, _class_names = load_model(model_dir)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print(f"Model loaded from {model_dir}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    return _model, _class_names


def process_image(image, model_dir):
    """
    Process an image and get predictions.
    
    Args:
        image: PIL Image
        model_dir: Directory containing the model
        
    Returns:
        Dictionary with prediction results
    """
    # Load the model
    model, class_names = load_model_singleton(model_dir)
    
    if model is None or class_names is None:
        raise RuntimeError("Failed to load model")
    
    # Get the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Make the prediction
    result = model.predict(image, class_names, device)
    
    return result


def get_model_info(model_dir):
    """
    Get information about the model.
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        Dictionary with model information
    """
    # Load the model
    model, class_names = load_model_singleton(model_dir)
    
    if model is None or class_names is None:
        raise RuntimeError("Failed to load model")
    
    # Get the model type
    model_path = os.path.join(model_dir, 'pokemon_card_classifier.pth')
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model type
    model_type = checkpoint.get('model_type', 'resnet50')  # Default to resnet50 for original model
    
    # Return the model information
    return {
        "model_type": model_type,
        "num_classes": len(class_names),
        "class_names": class_names
    }


def clear_model_cache():
    """
    Clear the model cache.
    """
    global _model, _class_names
    _model = None
    _class_names = None
    load_model_singleton.cache_clear()
