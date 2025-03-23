#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create example images for the web interface.
"""

import os
from PIL import Image, ImageDraw, ImageFont

# Create the examples directory if it doesn't exist
os.makedirs('static/examples', exist_ok=True)

# Define the example images
examples = [
    {
        'name': 'pikachu',
        'color': (255, 255, 0),  # Yellow
        'text_color': (0, 0, 0),  # Black
        'path': 'static/examples/pikachu.jpg'
    },
    {
        'name': 'charizard',
        'color': (255, 102, 0),  # Orange
        'text_color': (255, 255, 255),  # White
        'path': 'static/examples/charizard.jpg'
    },
    {
        'name': 'eevee',
        'color': (204, 153, 102),  # Brown
        'text_color': (255, 255, 255),  # White
        'path': 'static/examples/eevee.jpg'
    }
]

# Create and save the example images
for example in examples:
    print(f"Creating {example['name']} image...")
    
    try:
        # Create a new image
        image = Image.new('RGB', (300, 420), example['color'])
        
        # Get a drawing context
        draw = ImageDraw.Draw(image)
        
        # Try to get a font (use default if not available)
        try:
            font = ImageFont.truetype('arial.ttf', 36)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw the text
        text = f"{example['name'].capitalize()} Card"
        text_width = draw.textlength(text, font=font)
        text_position = ((300 - text_width) // 2, 210 - 18)
        draw.text(text_position, text, fill=example['text_color'], font=font)
        
        # Save the image
        image.save(example['path'])
        
        print(f"Saved {example['path']}")
    except Exception as e:
        print(f"Error creating {example['name']} image: {e}")

print("Done!")
