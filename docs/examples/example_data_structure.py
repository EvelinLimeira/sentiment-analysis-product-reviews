"""
Example script demonstrating the new data folder structure.

This script shows how to:
1. Load data with DataLoader
2. Save splits to the organized folder structure
3. Load splits from disk
4. Save processed data
5. Save perturbed test data
"""

from src.data_loader import DataLoader
import pandas as pd

def main():
    print("=" * 60)
    print("Data Structure Example")
    print("=" * 60)
    
    # Initialize DataLoader with a specific seed
    loader = DataLoader(
        dataset_name='amazon_reviews',
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    print("\n1. Loading dataset and creating splits...")
    train_df, val_df, test_df = loader.load()
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Save splits to organized folder structure
    print("\n2. Saving splits to data/raw/[train|validation|test]/...")
    loader.save_splits(format='csv')
    
    print("\n3. Folder structure created:")
    print("  data/")
    print("  ├── raw/")
    print("  │   ├── train/")
    print("  │   │   └── train_seed42.csv")
    print("  │   ├── validation/")
    print("  │   │   └── validation_seed42.csv")
    print("  │   └── test/")
    print("  │       └── test_seed42.csv")
    print("  ├── processed/")
    print("  │   ├── train/")
    print("  │   ├── validation/")
    print("  │   └── test/")
    print("  └── perturbed/")
    print("      └── test/")
    
    # Example: Save processed data (after preprocessing)
    print("\n4. Example: Saving processed data...")
    # In real usage, these would be preprocessed DataFrames
    loader.save_processed_splits(
        train_processed=train_df.copy(),
        val_processed=val_df.copy(),
        test_processed=test_df.copy(),
        suffix='bow',  # e.g., 'bow', 'embeddings', 'bert'
        format='csv'
    )
    print("  Saved to data/processed/[train|validation|test]/")
    
    # Example: Save perturbed test data
    print("\n5. Example: Saving perturbed test data...")
    # In real usage, this would be a perturbed version of test_df
    perturbed_test = test_df.copy()
    loader.save_perturbed_test(
        perturbed_df=perturbed_test,
        perturbation_type='typos',
        format='csv'
    )
    print("  Saved to data/perturbed/test/")
    
    # Example: Load splits from disk (useful for multiple simulations)
    print("\n6. Example: Loading splits from disk...")
    loader2 = DataLoader(random_state=42)
    train_loaded, val_loaded, test_loaded = loader2.load_splits(format='csv')
    print(f"  Loaded {len(train_loaded)} train, {len(val_loaded)} val, {len(test_loaded)} test samples")
    
    print("\n" + "=" * 60)
    print("Benefits of this structure:")
    print("  ✓ Prevents data leakage (physical separation)")
    print("  ✓ Easy to manage multiple simulations (seed-based filenames)")
    print("  ✓ Clear organization (raw → processed → perturbed)")
    print("  ✓ Reproducible experiments")
    print("=" * 60)

if __name__ == "__main__":
    main()
