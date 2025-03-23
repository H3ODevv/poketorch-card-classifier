#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script for training the improved Pokémon card classifier.
"""

import os
import argparse
import torch
from src.models.improved_train import main as train_main


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train the improved Pokémon card classifier')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='models/improved_pokemon_card_classifier',
                        help='Directory to save the model')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='efficientnet_b0',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet_b0', 'efficientnet_b3', 'convnext_tiny'],
                        help='Type of model to use')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and validation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Size of the input images')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=7,
                        help='Patience for early stopping')
    
    # Loss function arguments
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--cosine_t0', type=int, default=10,
                        help='T_0 parameter for CosineAnnealingWarmRestarts')
    parser.add_argument('--cosine_t_mult', type=int, default=2,
                        help='T_mult parameter for CosineAnnealingWarmRestarts')
    
    # Hardware arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training if available')
    
    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Check if CUDA is available
    if args.use_gpu and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU instead.")
    
    # Print GPU information if available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Create the save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Train the model
    train_main(args)


if __name__ == '__main__':
    main()
