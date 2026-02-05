"""
Download and test dataset for sentiment analysis.

This script downloads a dataset and tests the data loading pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


def main():
    print("=" * 60)
    print("Dataset Download and Test")
    print("=" * 60)
    print()
    
    # Initialize DataLoader
    print("Initializing DataLoader...")
    loader = DataLoader(
        dataset_name='amazon_reviews',
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    print("\nDownloading and loading dataset...")
    print("(This may take a few minutes on first run)")
    print()
    
    try:
        # Load data - this will download if needed
        train_df, val_df, test_df = loader.load()
        
        print("\n" + "=" * 60)
        print("SUCCESS! Dataset loaded")
        print("=" * 60)
        
        # Show statistics
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(train_df) + len(val_df) + len(test_df)}")
        print(f"  Train: {len(train_df)} samples ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        print(f"  Validation: {len(val_df)} samples ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        print(f"  Test: {len(test_df)} samples ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
        
        # Show class distribution
        print(f"\nClass Distribution:")
        distribution = loader.get_class_distribution()
        for split_name, counts in distribution.items():
            total = counts['negative'] + counts['positive']
            print(f"  {split_name.capitalize()}:")
            print(f"    Negative: {counts['negative']} ({counts['negative']/total*100:.1f}%)")
            print(f"    Positive: {counts['positive']} ({counts['positive']/total*100:.1f}%)")
        
        # Show sample reviews
        print(f"\nSample Reviews (Train):")
        samples = train_df.head(3)
        for idx, row in samples.iterrows():
            label = "Positive" if row['label'] == 1 else "Negative"
            text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print(f"  [{label}] {text_preview}")
        
        # Save splits
        print(f"\nSaving splits to organized folder structure...")
        loader.save_splits(format='csv')
        
        print(f"\nFiles saved to:")
        print(f"  data/raw/train/train_seed42.csv")
        print(f"  data/raw/validation/validation_seed42.csv")
        print(f"  data/raw/test/test_seed42.csv")
        
        print("\n" + "=" * 60)
        print("Dataset ready for experiments!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Failed to load dataset")
        print("=" * 60)
        print(f"\nError details: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection")
        print("  2. Install required packages: pip install datasets")
        print("  3. Try running again (downloads may be cached)")
        return 1


if __name__ == "__main__":
    exit(main())
