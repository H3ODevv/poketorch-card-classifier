#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to run the dataset preparation with the correct Python path.
"""

import os
import sys
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the prepare_dataset function
from src.data.prepare import prepare_dataset

def main():
    parser = argparse.ArgumentParser(description='Prepare the Pokémon card dataset for training')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--images', type=str, required=True, help='Directory containing the downloaded images')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save the processed dataset')
    parser.add_argument('--min-images', type=int, default=20, help='Minimum number of images required for a Pokémon')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of images to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Run the prepare_dataset function
    prepare_dataset(args.csv, args.images, args.output, args.min_images, args.train_ratio, args.seed)

if __name__ == '__main__':
    main()
