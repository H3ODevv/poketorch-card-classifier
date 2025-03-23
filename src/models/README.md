# PokéTorch Model Architecture and Training

This directory contains the model architecture and training scripts for the PokéTorch Pokémon card classifier.

## Model Architecture

The PokéTorch classifier uses transfer learning with pre-trained models to classify Pokémon cards. Two main model architectures are available:

### Basic Model (PokemonCardClassifier)

The basic model uses ResNet as the backbone with a simple classifier head:

```
ResNet (18/34/50/101) → Dropout(0.5) → Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, num_classes)
```

This model is defined in `model.py` and is trained using the script in `train.py`.

### Improved Model (ImprovedPokemonCardClassifier)

The improved model supports multiple backbone architectures with an enhanced classifier head:

```
Backbone (ResNet/EfficientNet/ConvNeXt) → Dropout(0.5) → Linear → BatchNorm → ReLU → Dropout(0.3) → Linear → BatchNorm → ReLU → Dropout(0.2) → Linear
```

This model is defined in `improved_train.py` and includes several improvements:

1. Support for modern architectures:

   - ResNet (18/34/50/101)
   - EfficientNet (B0/B3)
   - ConvNeXt Tiny

2. Enhanced classifier head with batch normalization for better training stability

## Training Process

### Basic Training

The basic training process is implemented in `train.py` and includes:

1. Data loading with basic augmentation:

   - Resize to 224x224
   - Random horizontal flip
   - Random rotation (±10°)
   - Color jitter (brightness, contrast, saturation)

2. Training with:
   - Cross-entropy loss
   - Adam optimizer
   - ReduceLROnPlateau learning rate scheduler
   - Early stopping

### Improved Training

The improved training process in `improved_train.py` includes several enhancements:

1. Enhanced data augmentation:

   - Resize to 224x224
   - Random horizontal flip
   - Random rotation (±30°)
   - Random affine transformations (shear, scale)
   - Enhanced color jitter (brightness, contrast, saturation, hue)
   - Random perspective changes

2. Class balancing techniques:

   - Weighted random sampling to balance class frequencies
   - Class weights in loss function

3. Advanced training techniques:
   - Focal Loss for handling hard examples and class imbalance
   - AdamW optimizer with weight decay for better regularization
   - Cosine Annealing with Warm Restarts for learning rate scheduling
   - Enhanced early stopping

## Model Evaluation

Both training scripts include comprehensive evaluation:

1. Training and validation metrics:

   - Loss
   - Accuracy

2. Visualization:

   - Loss curves
   - Accuracy curves
   - Confusion matrix

3. Classification report:
   - Per-class precision, recall, and F1-score
   - Macro and weighted averages

## Model Saving and Loading

Models are saved with their configuration and weights, allowing them to be easily loaded for inference:

```python
# Save model
save_model(model, save_dir, class_names)

# Load model
model, class_names = load_model(save_dir)
```

## Inference

The model includes methods for inference on new images:

```python
# Predict from a file path
result = predict_image(image_path, model, class_names)

# Predict from a PIL Image
result = model.predict(image, class_names)
```

The prediction result includes:

- The predicted class
- Confidence score
- Top 5 predictions with probabilities

## Performance Considerations

For best performance:

1. Use the improved training script with:

   - EfficientNet-B0 or ConvNeXt Tiny for a good balance of accuracy and speed
   - Focal Loss for handling class imbalance
   - Cosine Annealing scheduler for better convergence

2. Train with GPU acceleration:

   ```bash
   python train_improved_model.py --model_type efficientnet_b0 --use_focal_loss --use_gpu
   ```

3. For deployment, consider:
   - Using a smaller model like EfficientNet-B0 for faster inference
   - Quantizing the model for reduced memory footprint
   - Batch processing for multiple images
