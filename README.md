# PokéTorch Card Classifier

A deep learning-based Pokémon card classifier built with PyTorch. This project allows you to identify Pokémon characters from card images using a trained convolutional neural network.

## Features

- **Data Processing**: Scripts for downloading and preparing Pokémon card images
- **Model Training**: Training scripts for both basic and improved models
- **API**: FastAPI-based REST API for serving predictions
- **Web Interface**: User-friendly web interface for uploading and classifying images
- **Transfer Learning**: Utilizes pre-trained models (ResNet, EfficientNet, ConvNeXt) for better performance
- **Class Balancing**: Handles imbalanced datasets with weighted sampling and focal loss

## Project Structure

```
poketorch-card-classifier/
├── data/                      # Data directory
│   ├── raw/                   # Raw downloaded images
│   ├── processed/             # Processed images for training
│   └── metadata/              # Metadata files
├── models/                    # Trained models
│   ├── pokemon_card_classifier/       # Basic model
│   └── improved_pokemon_card_classifier/ # Improved model
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── api/                   # API code
│   ├── data/                  # Data processing code
│   └── models/                # Model definitions and training code
├── static/                    # Web interface static files
│   ├── css/                   # CSS styles
│   ├── js/                    # JavaScript code
│   └── examples/              # Example images
├── prepare_dataset.py         # Script to prepare the dataset
├── train_model.py             # Script to train the basic model
├── train_improved_model.py    # Script to train the improved model
├── run_api.py                 # Script to run the API
└── requirements.txt           # Python dependencies
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/poketorch-card-classifier.git
   cd poketorch-card-classifier
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Data Preparation

1. Download the Pokémon card dataset:

   ```bash
   python src/data/download.py
   ```

2. Prepare the dataset for training:
   ```bash
   python prepare_dataset.py
   ```

## Model Training

### Basic Model

Train the basic model with ResNet backbone:

```bash
python train_model.py --model_type resnet50 --num_epochs 30
```

### Improved Model

Train the improved model with EfficientNet backbone and focal loss:

```bash
python train_improved_model.py --model_type efficientnet_b0 --use_focal_loss --num_epochs 50
```

Available model types:

- `resnet18`, `resnet34`, `resnet50`, `resnet101`
- `efficientnet_b0`, `efficientnet_b3`
- `convnext_tiny`

## Running the API

Start the FastAPI server:

```bash
python run_api.py
```

The API will be available at http://localhost:8000

### API Endpoints

- `GET /health`: Health check endpoint
- `GET /info`: Get model information
- `GET /classes`: Get class names
- `POST /predict`: Predict Pokémon from an image
- `POST /predict/batch`: Predict Pokémon from multiple images
- `POST /reload`: Reload the model

## Web Interface

The web interface is served by the API server at http://localhost:8000

Features:

- Drag and drop image upload
- Real-time predictions
- Top 5 prediction results
- Example images for testing

## Model Architecture

### Basic Model

The basic model uses ResNet as the backbone with a simple classifier head:

```
ResNet → Dropout(0.5) → Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, num_classes)
```

### Improved Model

The improved model supports multiple backbone architectures with an enhanced classifier head:

```
Backbone → Dropout(0.5) → Linear → BatchNorm → ReLU → Dropout(0.3) → Linear → BatchNorm → ReLU → Dropout(0.2) → Linear
```

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Pokémon card dataset
- PyTorch and torchvision for the deep learning framework
- FastAPI for the API framework
