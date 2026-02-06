# Quick Start: Using the New Data Structure

## Overview

Your data is now organized in a clean folder structure that prevents data leakage and supports multiple simulations.

## Folder Structure

```
data/
├── raw/          # Original splits (after label conversion, before preprocessing)
│   ├── train/
│   ├── validation/
│   └── test/
├── processed/    # Preprocessed data ready for models
│   ├── train/
│   ├── validation/
│   └── test/
└── perturbed/    # Perturbed test data for robustness analysis
    └── test/
```

## Basic Usage

### 1. Load Data and Create Splits

```python
from src.data_loader import DataLoader

# Initialize with a seed for reproducibility
loader = DataLoader(random_state=42)

# Load dataset and create train/val/test splits
train_df, val_df, test_df = loader.load()

# Save splits to disk
loader.save_splits(format='csv')
```

This creates:
- `data/raw/train/train_seed42.csv`
- `data/raw/validation/validation_seed42.csv`
- `data/raw/test/test_seed42.csv`

### 2. Save Preprocessed Data

After preprocessing your data for a specific model:

```python
# After preprocessing (e.g., for Bag of Words)
loader.save_processed_splits(
    train_processed=train_bow,
    val_processed=val_bow,
    test_processed=test_bow,
    suffix='bow',
    format='csv'
)
```

This creates:
- `data/processed/train/train_bow_seed42.csv`
- `data/processed/validation/validation_bow_seed42.csv`
- `data/processed/test/test_bow_seed42.csv`

### 3. Save Perturbed Test Data

For robustness analysis:

```python
# After creating perturbed version
loader.save_perturbed_test(
    perturbed_df=test_with_typos,
    perturbation_type='typos',
    format='csv'
)
```

This creates:
- `data/perturbed/test/test_typos_seed42.csv`

### 4. Load Existing Splits

To load previously saved splits (useful for continuing experiments):

```python
loader = DataLoader(random_state=42)
train_df, val_df, test_df = loader.load_splits(format='csv')
```

## Multiple Simulations

For statistical validation with 10 different seeds (42-51):

```python
seeds = range(42, 52)  # Seeds 42-51 for 10 simulations

for seed in seeds:
    loader = DataLoader(random_state=seed)
    train, val, test = loader.load()
    loader.save_splits(format='csv')
    
    # Train your model...
    # Save processed data...
```

This creates separate files for each seed:
- `train_seed42.csv`, `train_seed43.csv`, ..., `train_seed51.csv`

**Note:** 10 simulations provide sufficient statistical power for validation with Wilcoxon and Kruskal-Wallis tests.

## File Naming Convention

All files follow this pattern:
```
{split}_{suffix}_seed{N}.{format}
```

Where:
- `{split}`: train, validation, or test
- `{suffix}`: Optional (e.g., bow, embeddings, bert, typos)
- `{N}`: Random seed number
- `{format}`: csv or parquet

Examples:
- `train_seed42.csv` - Raw training data with seed 42
- `test_bow_seed123.csv` - BoW preprocessed test data with seed 123
- `test_typos_seed42.csv` - Perturbed test data with typos, seed 42

## Benefits

- **No Data Leakage**: Physical separation prevents accidental mixing
- **Reproducible**: Seed-based filenames track exact splits
- **Organized**: Clear flow from raw → processed → perturbed
- **Scalable**: Easy to manage 10 simulations with different data splits
- **Statistical Validity**: 10 simulations provide sufficient power for significance testing

## Tips

1. **Always use the same seed** when loading/saving related data
2. **Use descriptive suffixes** for processed data (bow, embeddings, bert)
3. **Use CSV for inspection**, Parquet for performance
4. **Keep raw data intact** - never modify files in `data/raw/`

## Verification

Run this to verify your folder structure:

```bash
python test_folder_structure.py
```

## Full Example

See `example_data_structure.py` for a complete working example.
