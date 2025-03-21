#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare the Pokémon card dataset for training.
"""

import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
from collections import Counter
import random
import matplotlib.pyplot as plt
import re

def sanitize_filename(name):
    """
    Sanitize a string to be used as a directory name.
    Replaces invalid characters with underscores.
    
    Args:
        name: The string to sanitize
        
    Returns:
        A sanitized string that can be used as a directory name
    """
    # Replace characters that are not allowed in directory names
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def prepare_dataset(csv_path, images_dir, output_dir, min_images=20, train_ratio=0.8, seed=42):
    """
    Prepare the dataset for training by organizing images by Pokémon name.
    
    Args:
        csv_path: Path to the CSV file
        images_dir: Directory containing the downloaded images
        output_dir: Directory to save the processed dataset
        min_images: Minimum number of images required for a Pokémon to be included
        train_ratio: Ratio of images to use for training
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with statistics about the dataset
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} cards in the CSV file")
    
    # Count the number of images per Pokémon
    pokemon_counts = Counter(df['name'])
    
    # Filter Pokémon with enough images
    valid_pokemon = [name for name, count in pokemon_counts.items() if count >= min_images]
    print(f"Found {len(valid_pokemon)} Pokémon with at least {min_images} images")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create directories for each Pokémon
    pokemon_dir_map = {}  # Map from original name to sanitized directory name
    for pokemon in valid_pokemon:
        # Sanitize the Pokémon name for use as a directory name
        sanitized_name = sanitize_filename(pokemon)
        pokemon_dir_map[pokemon] = sanitized_name
        
        # Create directories
        os.makedirs(os.path.join(train_dir, sanitized_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, sanitized_name), exist_ok=True)
    
    # Process each Pokémon
    stats = {'pokemon': [], 'train': [], 'val': []}
    
    for pokemon in tqdm(valid_pokemon, desc="Processing Pokémon"):
        # Get the sanitized directory name
        dir_name = pokemon_dir_map[pokemon]
        
        # Get all cards for this Pokémon
        pokemon_df = df[df['name'] == pokemon]
        
        # Get the image IDs
        image_ids = pokemon_df['id'].tolist()
        
        # Shuffle the image IDs
        random.shuffle(image_ids)
        
        # Split into train and validation sets
        split_idx = int(len(image_ids) * train_ratio)
        train_ids = image_ids[:split_idx]
        val_ids = image_ids[split_idx:]
        
        # Copy images to the appropriate directories
        for image_id in train_ids:
            src = os.path.join(images_dir, f"{image_id}.jpg")
            dst = os.path.join(train_dir, dir_name, f"{image_id}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        for image_id in val_ids:
            src = os.path.join(images_dir, f"{image_id}.jpg")
            dst = os.path.join(val_dir, dir_name, f"{image_id}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
        
        # Count the actual number of images copied
        train_count = len([f for f in os.listdir(os.path.join(train_dir, dir_name)) if f.endswith('.jpg')])
        val_count = len([f for f in os.listdir(os.path.join(val_dir, dir_name)) if f.endswith('.jpg')])
        
        # Add to statistics
        stats['pokemon'].append(pokemon)
        stats['train'].append(train_count)
        stats['val'].append(val_count)
    
    # Convert to DataFrame for easier analysis
    stats_df = pd.DataFrame(stats)
    stats_df['total'] = stats_df['train'] + stats_df['val']
    stats_df = stats_df.sort_values('total', ascending=False)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total Pokémon: {len(stats_df)}")
    print(f"Total Training Images: {stats_df['train'].sum()}")
    print(f"Total Validation Images: {stats_df['val'].sum()}")
    
    # Save statistics to CSV
    stats_df.to_csv(os.path.join(output_dir, 'dataset_stats.csv'), index=False)
    
    # Plot the distribution of images
    plt.figure(figsize=(12, 6))
    plt.bar(stats_df['pokemon'][:10], stats_df['total'][:10])
    plt.title('Number of Images per Pokémon (Top 10)')
    plt.xlabel('Pokémon')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'))
    
    # Save the mapping from original names to directory names
    with open(os.path.join(output_dir, 'pokemon_name_mapping.csv'), 'w', encoding='utf-8') as f:
        f.write('original_name,directory_name\n')
        for pokemon, dir_name in pokemon_dir_map.items():
            f.write(f'"{pokemon}","{dir_name}"\n')
    
    return stats_df

def main():
    parser = argparse.ArgumentParser(description='Prepare the Pokémon card dataset for training')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--images', type=str, required=True, help='Directory containing the downloaded images')
    parser.add_argument('--output', type=str, default='data/processed', help='Directory to save the processed dataset')
    parser.add_argument('--min-images', type=int, default=20, help='Minimum number of images required for a Pokémon')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of images to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    prepare_dataset(args.csv, args.images, args.output, args.min_images, args.train_ratio, args.seed)

if __name__ == '__main__':
    main()
