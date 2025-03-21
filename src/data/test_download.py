#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the download functionality with a small number of images.
"""

import os
import argparse
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.download import download_all_images

def main():
    parser = argparse.ArgumentParser(description='Test downloading a small number of Pokémon card images')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--output', type=str, default='data/raw_test', help='Directory to save the test images')
    parser.add_argument('--count', type=int, default=10, help='Number of images to download for testing')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Testing download with {args.count} images...")
    download_all_images(args.csv, args.output, max_workers=5, limit=args.count)
    print("Test completed!")

if __name__ == '__main__':
    main()
