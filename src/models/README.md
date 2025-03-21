# Pokémon Card Classifier Model

This directory contains the model architecture, training, and utility scripts for the Pokémon card classifier.

## Model Architecture

The model architecture is defined in `model.py`. It uses transfer learning with a ResNet backbone (ResNet18, ResNet34, ResNet50, or ResNet101) and a custom classification head.

### Key Components

- `PokemonCardClassifier`: The main model class that uses a ResNet backbone with a custom classification head
- `save_model`: Function to save the model and class names
- `load_model`: Function to load the model and class names
- `predict_image`: Function to predict the class of an image
- `DeviceDataLoader`: Wrapper for DataLoader to move data to the device (CPU or GPU)

## Training

The training script is defined in `train.py`. It handles loading the dataset, training the model, and evaluating its performance.

### Key Components

- `get_transforms`: Function to get the transforms for training and validation
- `load_data`: Function to load the dataset
- `train_epoch`: Function to train the model for one epoch
- `validate`: Function to validate the model
- `train`: Function to train the model for multiple epochs with early stopping
- `plot_metrics`: Function to plot the training and validation metrics
- `plot_confusion_matrix`: Function to plot the confusion matrix

## Utilities

The utility functions are defined in `utils.py`. They provide additional functionality for model evaluation and visualization.

### Key Components

- `evaluate_model`: Function to evaluate the model on a dataset
- `visualize_predictions`: Function to visualize model predictions on a few samples
- `predict_image_from_path`: Function to predict the class of an image from a file path
- `predict_image_from_array`: Function to predict the class of an image from a numpy array
- `batch_predict`: Function to predict the classes of multiple images
- `export_model_info`: Function to export model information to a JSON file
- `create_model_summary`: Function to create a summary of the model

## Usage

### Training the Model

To train the model, use the `train_model.py` script at the project root:

```bash
python train_model.py --data_dir data/processed --model_type resnet50 --batch_size 32 --num_epochs 30
```

#### Arguments

- `--data_dir`: Directory containing the dataset (default: 'data/processed')
- `--model_type`: Type of ResNet model to use (choices: 'resnet18', 'resnet34', 'resnet50', 'resnet101', default: 'resnet50')
- `--save_dir`: Directory to save the model (default: 'models/pokemon_card_classifier')
- `--batch_size`: Batch size for training and validation (default: 32)
- `--img_size`: Size of the input images (default: 224)
- `--num_epochs`: Number of epochs to train for (default: 30)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0001)
- `--patience`: Patience for early stopping (default: 5)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--use_gpu`: Use GPU for training if available (flag)

### Using the Trained Model

To use the trained model for prediction, you can use the utility functions in `utils.py`:

```python
from src.models.utils import predict_image_from_path

# Predict the class of an image
result = predict_image_from_path(
    image_path='path/to/image.jpg',
    model_dir='models/pokemon_card_classifier'
)

# Print the prediction
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Evaluating the Model

To evaluate the model on a dataset, you can use the `evaluate_model` function in `utils.py`:

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.models.model import load_model, DeviceDataLoader
from src.models.utils import evaluate_model

# Load the model
model, class_names = load_model('models/pokemon_card_classifier')

# Load the dataset
val_dataset = datasets.ImageFolder(
    'data/processed/val',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

# Create the dataloader
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Wrap the dataloader to move data to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_loader = DeviceDataLoader(val_loader, device)

# Evaluate the model
results = evaluate_model(model, val_loader, device)

# Print the results
print(f"Accuracy: {results['accuracy']:.4f}")
```

## Model Outputs

After training, the model will save the following files in the specified save directory:

- `pokemon_card_classifier.pth`: The trained model weights
- `class_names.json`: The class names
- `loss.png`: Plot of the training and validation loss
- `accuracy.png`: Plot of the training and validation accuracy
- `confusion_matrix.png`: Confusion matrix
- `classification_report.json`: Classification report with precision, recall, and F1-score for each class
