# Data Flow Diagram

## Overview

This diagram shows how data flows through the folder structure during the sentiment analysis pipeline.

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCE                          │
│         (Hugging Face datasets, CSV files, etc.)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ DataLoader.load()
                             │ - Extract text & rating
                             │ - Convert to binary labels
                             │ - Discard neutral (rating 3)
                             │ - Stratified split (70/15/15)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      data/raw/                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    train/    │  │ validation/  │  │    test/     │          │
│  │              │  │              │  │              │          │
│  │ train_seed   │  │ validation_  │  │ test_seed    │          │
│  │   42.csv     │  │ seed42.csv   │  │   42.csv     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Original splits with text and binary labels                    │
│  NO preprocessing applied yet                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Preprocessor.fit_transform()
                             │ - Lowercase
                             │ - Remove special chars
                             │ - Remove stopwords
                             │ - Tokenization
                             │
                             │ BoW_Vectorizer / Embedding_Encoder
                             │ - TF-IDF vectors
                             │ - Word embeddings
                             │ - BERT tokenization
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    data/processed/                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    train/    │  │ validation/  │  │    test/     │          │
│  │              │  │              │  │              │          │
│  │ train_bow_   │  │ validation_  │  │ test_bow_    │          │
│  │ seed42.csv   │  │ bow_seed42   │  │ seed42.csv   │          │
│  │              │  │              │  │              │          │
│  │ train_emb_   │  │ validation_  │  │ test_emb_    │          │
│  │ seed42.csv   │  │ emb_seed42   │  │ seed42.csv   │          │
│  │              │  │              │  │              │          │
│  │ train_bert_  │  │ validation_  │  │ test_bert_   │          │
│  │ seed42.csv   │  │ bert_seed42  │  │ seed42.csv   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Preprocessed data ready for model training                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ For robustness analysis
                             │ 
                             │
                             │ AdvancedAnalysis.create_perturbed()
                             │ - Add typos (5% char swaps)
                             │ - Remove accents
                             │ - Duplicate letters
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    data/perturbed/                               │
│                    ┌──────────────┐                              │
│                    │    test/     │                              │
│                    │              │                              │
│                    │ test_typos_  │                              │
│                    │ seed42.csv   │                              │
│                    │              │                              │
│                    │ test_emojis_ │                              │
│                    │ seed42.csv   │                              │
│                    │              │                              │
│                    │ test_formal_ │                              │
│                    │ seed42.csv   │                              │
│                    └──────────────┘                              │
│                                                                  │
│  Perturbed test data for robustness evaluation                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Principles

### 1. One-Way Flow
```
External Source → raw → processed → perturbed
```
Data flows in one direction. Never modify files in earlier stages.

### 2. Seed-Based Naming
```
{split}_{suffix}_seed{N}.{format}
```
Every file is tagged with its random seed for reproducibility.

### 3. Physical Separation
```
train/     ← Training data (70%)
validation/ ← Validation data (15%)
test/      ← Test data (15%)
```
Separate folders prevent data leakage.

## Multiple Simulations

For statistical validation (Requirement 7.1), run with different seeds:

```
Seed 42:
  data/raw/train/train_seed42.csv
  data/raw/validation/validation_seed42.csv
  data/raw/test/test_seed42.csv

Seed 123:
  data/raw/train/train_seed123.csv
  data/raw/validation/validation_seed123.csv
  data/raw/test/test_seed123.csv

Seed 456:
  data/raw/train/train_seed456.csv
  ...
```

Each simulation has its own set of files, making it easy to:
- Track which results came from which split
- Reproduce exact experiments
- Run simulations in parallel
- Aggregate results for statistical tests

## Model-Specific Processing

Different models need different preprocessing:

```
data/processed/train/
├── train_bow_seed42.csv      ← SVM + BoW (tokenized, TF-IDF)
├── train_embeddings_seed42.csv ← SVM + Embeddings (Word2Vec/GloVe)
└── train_bert_seed42.csv     ← BERT (minimal preprocessing)
```

Each model gets its own processed version, avoiding conflicts.

## Robustness Analysis

Test data can be perturbed in multiple ways:

```
data/perturbed/test/
├── test_typos_seed42.csv     ← Character swaps, accent removal
├── test_emojis_seed42.csv    ← With/without emojis
└── test_formality_seed42.csv ← Different formality levels
```

Compare model accuracy on clean vs. perturbed data to measure robustness.

## Code Example

```python
from src.data_loader import DataLoader

# Load and split
loader = DataLoader(random_state=42)
train, val, test = loader.load()

# Save raw splits
loader.save_splits()
# → data/raw/[train|validation|test]/..._seed42.csv

# Preprocess for BoW
train_bow = preprocess_bow(train)
val_bow = preprocess_bow(val)
test_bow = preprocess_bow(test)

# Save processed
loader.save_processed_splits(train_bow, val_bow, test_bow, suffix='bow')
# → data/processed/[train|validation|test]/..._bow_seed42.csv

# Create perturbed test
test_typos = add_typos(test)

# Save perturbed
loader.save_perturbed_test(test_typos, perturbation_type='typos')
# → data/perturbed/test/test_typos_seed42.csv
```

## Benefits Summary

| Benefit | How It's Achieved |
|---------|-------------------|
| No Data Leakage | Physical separation in folders |
| Reproducibility | Seed-based filenames |
| Multiple Simulations | One file per seed |
| Clear Organization | Intuitive folder hierarchy |
| Model Flexibility | Suffix-based naming |
| Statistical Validation | Easy to aggregate results |

## Requirements Mapping

| Requirement | Implementation |
|-------------|----------------|
| 1.5: Split before preprocessing | `data/raw/` contains pre-preprocessing splits |
| 1.6: No data leakage | Separate `train/`, `validation/`, `test/` folders |
| 1.7: Class distribution | Tracked in DataLoader, saved with splits |
| 1.8: Multiple seeds | Seed-based filenames |
| 8.3: Perturbed test data | `data/perturbed/test/` folder |
| 7.1: Multiple simulations | One file per seed (10-30 simulations) |
