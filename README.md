# PokéTorch: Pokémon Card Classifier

A PyTorch-based image classifier for Pokémon cards using transfer learning.

## Project Overview

PokéTorch is a one-day project that demonstrates:

1. **Image Classification** with **PyTorch** and **Transfer Learning**
2. **End-to-End** ML workflow: data gathering, training, inference, and serving predictions via **FastAPI**
3. A bit of **fun** using _Pokémon card images_ as our classification target

## Project Structure

```
poketorch-card-classifier/
├── data/
│   ├── raw/                  # Downloaded card images
│   ├── processed/            # Organized train/val splits
│   └── metadata/             # CSV and other metadata
├── src/
│   ├── data/
│   │   ├── download.py       # Image downloading script
│   │   ├── prepare.py        # Dataset preparation
│   │   └── test_download.py  # Test script for downloading
│   ├── models/
│   │   ├── model.py          # Model architecture
│   │   └── train.py          # Training script
│   └── api/
│       ├── app.py            # FastAPI application
│       └── utils.py          # API utilities
├── notebooks/                # Exploratory analysis
├── models/                   # Saved model checkpoints
├── static/                   # Web interface assets
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Setup and Installation

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

## Data Processing

### 1. Test Download

Before downloading all images, you can test the download functionality with a small number of images:

```bash
python src/data/test_download.py --csv data/metadata/pokemon-cards.csv --count 10
```

### 2. Download Images

Download all Pokémon card images from the dataset:

```bash
python src/data/download.py --csv data/metadata/pokemon-cards.csv --output data/raw
```

### 3. Prepare Dataset

Organize the downloaded images by Pokémon name and create train/validation splits:

```bash
python src/data/prepare.py --csv data/metadata/pokemon-cards.csv --images data/raw --output data/processed
```

## Model Training

Train the model using the prepared dataset:

```bash
# Train with default settings (ResNet50)
python train_model.py --use_gpu

# Train with a different model architecture
python train_model.py --model_type resnet18 --use_gpu

# Train with custom parameters
python train_model.py --model_type resnet34 --batch_size 64 --learning_rate 0.0005 --num_epochs 50 --use_gpu
```

The training script will:

1. Load the processed dataset from `data/processed`
2. Train a ResNet-based model using transfer learning
3. Save the trained model, metrics, and visualizations to `models/pokemon_card_classifier`

For more details on the model architecture and training process, see [src/models/README.md](src/models/README.md).

## API Deployment

Run the FastAPI server to serve predictions:

```bash
# Coming soon
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset source: [Pokémon Cards Dataset on Kaggle](https://www.kaggle.com/datasets/priyamchoksi/pokemon-cards)
- Inspired by the Pokémon Trading Card Game
