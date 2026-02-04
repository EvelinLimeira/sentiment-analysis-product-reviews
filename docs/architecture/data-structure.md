# Data Folder Structure

This document describes the organized data folder structure for the sentiment analysis NLP project.

## Directory Structure

```
data/
├── raw/
│   ├── train/
│   │   └── train_seed{N}.csv
│   ├── validation/
│   │   └── validation_seed{N}.csv
│   └── test/
│       └── test_seed{N}.csv
├── processed/
│   ├── train/
│   │   ├── train_bow_seed{N}.csv
│   │   ├── train_embeddings_seed{N}.csv
│   │   └── train_bert_seed{N}.csv
│   ├── validation/
│   │   ├── validation_bow_seed{N}.csv
│   │   ├── validation_embeddings_seed{N}.csv
│   │   └── validation_bert_seed{N}.csv
│   └── test/
│       ├── test_bow_seed{N}.csv
│       ├── test_embeddings_seed{N}.csv
│       └── test_bert_seed{N}.csv
└── perturbed/
    └── test/
        ├── test_typos_seed{N}.csv
        ├── test_emojis_seed{N}.csv
        └── test_formality_seed{N}.csv
```

Where `{N}` is the random seed number (e.g., 42, 123, 456).

## Directory Purposes

### `data/raw/`
Contains the original split data after loading and label conversion, but before any preprocessing.

- **train/**: Training set (70% of data)
- **validation/**: Validation set (15% of data)
- **test/**: Test set (15% of data)

Files are named with the seed to support multiple simulations with different random splits.

### `data/processed/`
Contains preprocessed data ready for model training/evaluation.

Each subfolder contains processed versions for different models:
- **bow**: Preprocessed for Bag of Words (tokenized, stopwords removed)
- **embeddings**: Preprocessed for embeddings (tokenized, ready for Word2Vec/GloVe)
- **bert**: Minimal preprocessing (BERT uses its own tokenizer)

### `data/perturbed/`
Contains perturbed versions of the test set for robustness analysis.

- **typos**: Test data with character swaps, accent removal, letter duplication
- **emojis**: Test data with/without emojis for emoji impact analysis
- **formality**: Test data with different formality levels

## Benefits

1. **Prevents Data Leakage**: Physical separation of train/val/test ensures no accidental mixing
2. **Reproducibility**: Seed-based filenames allow exact reproduction of experiments
3. **Multiple Simulations**: Easy to manage 10-30 simulations with different seeds
4. **Clear Organization**: Raw → Processed → Perturbed flow is intuitive
5. **Model-Specific Processing**: Each model can have its own preprocessed version

## Usage Example

```python
from src.data_loader import DataLoader

# Initialize with seed
loader = DataLoader(random_state=42)

# Load and split data
train_df, val_df, test_df = loader.load()

# Save splits to organized structure
loader.save_splits(format='csv')
# Creates: data/raw/train/train_seed42.csv, etc.

# Save processed data for a specific model
loader.save_processed_splits(
    train_processed=train_bow,
    val_processed=val_bow,
    test_processed=test_bow,
    suffix='bow',
    format='csv'
)
# Creates: data/processed/train/train_bow_seed42.csv, etc.

# Save perturbed test data
loader.save_perturbed_test(
    perturbed_df=test_perturbed,
    perturbation_type='typos',
    format='csv'
)
# Creates: data/perturbed/test/test_typos_seed42.csv

# Load existing splits (useful for continuing experiments)
loader2 = DataLoader(random_state=42)
train, val, test = loader2.load_splits(format='csv')
```

## File Formats

Both CSV and Parquet formats are supported:

- **CSV**: Human-readable, easy to inspect, larger file size
- **Parquet**: Compressed, faster I/O, smaller file size

Choose based on your needs:
```python
loader.save_splits(format='csv')      # For inspection
loader.save_splits(format='parquet')  # For performance
```

## Requirements Alignment

This structure aligns with project requirements:

- **Requirement 1.5**: Stratified split BEFORE preprocessing ✓
- **Requirement 1.6**: No data leakage (physical separation) ✓
- **Requirement 1.7**: Class distribution tracking ✓
- **Requirement 1.8**: Multiple random seeds support ✓
- **Requirement 8.3**: Perturbed test data for robustness ✓

## Notes

- All directories are created automatically when saving data
- Seed numbers in filenames ensure you can track which split corresponds to which simulation
- The structure supports both single experiments and large-scale statistical validation (10-30 simulations)
