# Data Processing Scripts for PokéTorch

This directory contains scripts for downloading and preparing the Pokémon card dataset for training.

## Dataset Source

The Pokémon card dataset used in this project is from Kaggle:

- **Name**: Pokemon Cards
- **Author**: Priyam Choksi
- **URL**: [https://www.kaggle.com/datasets/priyamchoksi/pokemon-cards](https://www.kaggle.com/datasets/priyamchoksi/pokemon-cards)

Please cite the dataset appropriately if you use it in your work (see the main README.md for citation format).

## Test Download Script

The `test_download.py` script allows you to test the download functionality with a small number of images before downloading the entire dataset.

### Usage

```bash
python src/data/test_download.py --csv <path_to_csv> [--output <output_directory>] [--count <number_of_images>]
```

### Arguments

- `--csv`: Path to the CSV file containing card data (required)
- `--output`: Directory to save the test images (default: 'data/raw_test')
- `--count`: Number of images to download for testing (default: 10)

### Example

```bash
# Test downloading 10 images
python src/data/test_download.py --csv pokemon-cards.csv

# Test downloading 20 images to a custom directory
python src/data/test_download.py --csv pokemon-cards.csv --output data/test_images --count 20
```

## Download Script

The `download.py` script downloads Pokémon card images from the URLs provided in the CSV file.

### Usage

```bash
python src/data/download.py --csv <path_to_csv> --output <output_directory> [--workers <num_workers>] [--limit <limit>]
```

### Arguments

- `--csv`: Path to the CSV file containing card data (required)
- `--output`: Directory to save the downloaded images (default: 'data/raw')
- `--workers`: Maximum number of concurrent downloads (default: 10)
- `--limit`: Maximum number of images to download, useful for testing (optional)

### Example

```bash
# Download all images
python src/data/download.py --csv pokemon-cards.csv --output data/raw

# Download only 100 images for testing
python src/data/download.py --csv pokemon-cards.csv --output data/raw --limit 100
```

## Prepare Script

The `prepare.py` script organizes the downloaded images by Pokémon name and creates train/validation splits.

### Usage

```bash
python src/data/prepare.py --csv <path_to_csv> --images <images_directory> --output <output_directory> [--min-images <min_images>] [--train-ratio <train_ratio>] [--seed <seed>]
```

### Arguments

- `--csv`: Path to the CSV file containing card data (required)
- `--images`: Directory containing the downloaded images (required)
- `--output`: Directory to save the processed dataset (default: 'data/processed')
- `--min-images`: Minimum number of images required for a Pokémon to be included (default: 20)
- `--train-ratio`: Ratio of images to use for training (default: 0.8)
- `--seed`: Random seed for reproducibility (default: 42)

### Example

```bash
# Prepare the dataset with default settings
python src/data/prepare.py --csv pokemon-cards.csv --images data/raw --output data/processed

# Prepare the dataset with custom settings
python src/data/prepare.py --csv pokemon-cards.csv --images data/raw --output data/processed --min-images 30 --train-ratio 0.75
```

## Output

The prepare script will create the following directory structure:

```
data/processed/
├── train/
│   ├── pokemon1/
│   ├── pokemon2/
│   └── ...
├── val/
│   ├── pokemon1/
│   ├── pokemon2/
│   └── ...
├── dataset_stats.csv
└── dataset_distribution.png
```

- `train/`: Contains training images organized by Pokémon name
- `val/`: Contains validation images organized by Pokémon name
- `dataset_stats.csv`: Statistics about the dataset
- `dataset_distribution.png`: Plot showing the distribution of images per Pokémon
