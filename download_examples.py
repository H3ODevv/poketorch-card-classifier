#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download example images for the web interface.
"""

import os
import requests
from PIL import Image
from io import BytesIO

# Create the examples directory if it doesn't exist
os.makedirs('static/examples', exist_ok=True)

# Define the example images
examples = [
    {
        'name': 'pikachu',
        'url': 'https://via.placeholder.com/300x420/ffff00/000000?text=Pikachu+Card',
        'path': 'static/examples/pikachu.jpg'
    },
    {
        'name': 'charizard',
        'url': 'https://via.placeholder.com/300x420/ff6600/ffffff?text=Charizard+Card',
        'path': 'static/examples/charizard.jpg'
    },
    {
        'name': 'eevee',
        'url': 'https://via.placeholder.com/300x420/cc9966/ffffff?text=Eevee+Card',
        'path': 'static/examples/eevee.jpg'
    }
]

# Download and save the example images
for example in examples:
    print(f"Downloading {example['name']} image...")
    
    try:
        # Download the image
        response = requests.get(example['url'])
        response.raise_for_status()
        
        # Open the image
        image = Image.open(BytesIO(response.content))
        
        # Save the image
        image.save(example['path'])
        
        print(f"Saved {example['path']}")
    except Exception as e:
        print(f"Error downloading {example['name']} image: {e}")

print("Done!")
