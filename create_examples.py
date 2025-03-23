#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create example images for the web interface.
Creates placeholder images for Alakazam, Blastoise, and Chansey if they don't exist.
"""

import os
from PIL import Image, ImageDraw, ImageFont

# Create the examples directory if it doesn't exist
os.makedirs('static/examples', exist_ok=True)

# Define the example images
examples = [
    {
        'name': 'alakazam',
        'color': (255, 215, 0),  # Gold
        'text_color': (0, 0, 0),  # Black
        'path': 'static/examples/alakazam.jpg'
    },
    {
        'name': 'blastoise',
        'color': (0, 102, 204),  # Blue
        'text_color': (255, 255, 255),  # White
        'path': 'static/examples/blastoise.jpg'
    },
    {
        'name': 'chansey',
        'color': (255, 182, 193),  # Pink
        'text_color': (0, 0, 0),  # Black
        'path': 'static/examples/chansey.jpg'
    }
]

# Create placeholder images if they don't already exist
for example in examples:
    print(f"Processing {example['name']} image...")
    
    # Skip if the file already exists
    if os.path.exists(example['path']):
        print(f"Image for {example['name']} already exists, skipping...")
        continue
    
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
        
        print(f"Created placeholder for {example['name']}")
    except Exception as e:
        print(f"Error creating {example['name']} image: {e}")

print("Done!")
