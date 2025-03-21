#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download Pokémon card images from the dataset.
"""

import os
import csv
import requests
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

def download_image(row, output_dir, retry_count=3, delay=1):
    """
    Download an image from a URL and save it to the specified directory.
    
    Args:
        row: A tuple of (card_id, image_url, name)
        output_dir: Directory to save the image
        retry_count: Number of retries if download fails
        delay: Delay between retries in seconds
        
    Returns:
        Tuple of (success, card_id, name, error_message)
    """
    card_id, image_url, name = row
    
    # Create a filename based on the card ID
    filename = os.path.join(output_dir, f"{card_id}.jpg")
    
    # Skip if file already exists
    if os.path.exists(filename):
        return True, card_id, name, "Already exists"
    
    # Try to download the image
    for attempt in range(retry_count):
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            
            # Save the image
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return True, card_id, name, "Success"
            
        except Exception as e:
            error_message = str(e)
            if attempt < retry_count - 1:
                # Add a small random delay to avoid hammering the server
                time.sleep(delay + random.random())
                
    return False, card_id, name, error_message

def download_all_images(csv_path, output_dir, max_workers=10, limit=None):
    """
    Download all images from the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save the images
        max_workers: Maximum number of concurrent downloads
        limit: Maximum number of images to download (for testing)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} cards in the CSV file")
    
    # Limit the number of downloads if specified
    if limit:
        df = df.head(limit)
        print(f"Limiting to {limit} downloads for testing")
    
    # Prepare the download tasks
    tasks = [(row['id'], row['image_url'], row['name']) for _, row in df.iterrows()]
    
    # Download the images using multiple threads
    successful = 0
    failed = 0
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, task, output_dir): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            success, card_id, name, message = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                results.append((card_id, name, message))
    
    # Print summary
    print(f"\nDownload Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Write failed downloads to a file
    if failed > 0:
        with open(os.path.join(output_dir, "../failed_downloads.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['card_id', 'name', 'error'])
            writer.writerows(results)
        print(f"  Failed downloads saved to {os.path.join(output_dir, '../failed_downloads.csv')}")

def main():
    parser = argparse.ArgumentParser(description='Download Pokémon card images')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--output', type=str, default='data/raw', help='Directory to save the images')
    parser.add_argument('--workers', type=int, default=10, help='Maximum number of concurrent downloads')
    parser.add_argument('--limit', type=int, help='Maximum number of images to download (for testing)')
    
    args = parser.parse_args()
    
    download_all_images(args.csv, args.output, args.workers, args.limit)

if __name__ == '__main__':
    main()
