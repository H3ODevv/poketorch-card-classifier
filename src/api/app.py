#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FastAPI application for the Pokémon card classifier.
"""

import os
import io
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from src.api.utils import process_image, get_model_info, clear_model_cache

# Define the model directory
MODEL_DIR = os.environ.get("MODEL_DIR", "models/pokemon_card_classifier")
IMPROVED_MODEL_DIR = os.environ.get("IMPROVED_MODEL_DIR", "models/improved_pokemon_card_classifier")

# Check if the improved model exists
if os.path.exists(IMPROVED_MODEL_DIR) and os.path.isdir(IMPROVED_MODEL_DIR):
    # Use the improved model if it exists
    MODEL_DIR = IMPROVED_MODEL_DIR

# Create the FastAPI application
app = FastAPI(
    title="PokéTorch Card Classifier API",
    description="API for classifying Pokémon cards using a trained model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Define the prediction response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top5_classes: List[str]
    top5_probabilities: List[float]


# Define the batch prediction request model
class BatchPredictionRequest(BaseModel):
    image_urls: List[str]


# Define the health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {"status": "ok"}


# Define the model info endpoint
@app.get("/info")
async def model_info():
    """
    Get information about the model.
    
    Returns:
        dict: Model information
    """
    try:
        info = get_model_info(MODEL_DIR)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


# Define the class names endpoint
@app.get("/classes")
async def get_classes():
    """
    Get the class names.
    
    Returns:
        dict: Class names
    """
    try:
        info = get_model_info(MODEL_DIR)
        return {"class_names": info["class_names"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting class names: {str(e)}")


# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict the Pokémon in an image.
    
    Args:
        file: The image file
        
    Returns:
        dict: Prediction results
    """
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process the image
        result = process_image(image, MODEL_DIR)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Define the batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict the Pokémon in multiple images.
    
    Args:
        files: The image files
        
    Returns:
        dict: Prediction results
    """
    try:
        results = []
        
        for file in files:
            # Read the image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Process the image
            result = process_image(image, MODEL_DIR)
            results.append({
                "filename": file.filename,
                "prediction": result
            })
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")


# Define the reload model endpoint
@app.post("/reload")
async def reload_model():
    """
    Reload the model.
    
    Returns:
        dict: Reload status
    """
    try:
        clear_model_cache()
        return {"status": "ok", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


# Import RedirectResponse
from fastapi.responses import RedirectResponse

# Define the root endpoint
@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        RedirectResponse: Redirect to the web interface
    """
    return RedirectResponse(url="/static/index.html")

# Define the API info endpoint
@app.get("/api")
async def api_info():
    """
    API information endpoint.
    
    Returns:
        dict: API information
    """
    return {
        "name": "PokéTorch Card Classifier API",
        "version": "1.0.0",
        "description": "API for classifying Pokémon cards using a trained model",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "classes": "/classes",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "reload": "/reload"
        }
    }
