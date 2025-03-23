#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the FastAPI application for the Pokémon card classifier.
"""

import os
import argparse
import uvicorn

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the PokéTorch Card Classifier API')
    
    # Server arguments
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload for development')
    
    # Model arguments
    parser.add_argument('--model-dir', type=str, default='models/pokemon_card_classifier',
                        help='Directory containing the model')
    parser.add_argument('--improved-model-dir', type=str, default='models/improved_pokemon_card_classifier',
                        help='Directory containing the improved model')
    
    return parser.parse_args()

def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set environment variables
    os.environ['MODEL_DIR'] = args.model_dir
    os.environ['IMPROVED_MODEL_DIR'] = args.improved_model_dir
    
    # Print server information
    print(f"Starting PokéTorch Card Classifier API on http://{args.host}:{args.port}")
    print(f"Model directory: {args.model_dir}")
    print(f"Improved model directory: {args.improved_model_dir}")
    
    # Check if the improved model exists
    if os.path.exists(args.improved_model_dir) and os.path.isdir(args.improved_model_dir):
        print(f"Using improved model from {args.improved_model_dir}")
    else:
        print(f"Using original model from {args.model_dir}")
    
    # Run the server
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == '__main__':
    main()
