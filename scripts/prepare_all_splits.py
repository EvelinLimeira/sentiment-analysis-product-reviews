"""
Pre-generate all data splits for simulations.

This script creates and saves all 30 train/val/test splits with different
random seeds. This avoids re-downloading from Hugging Face during simulations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from tqdm import tqdm


def prepare_all_splits(num_simulations: int = 30, base_seed: int = 42):
    """
    Pre-generate and save all data splits.
    
    Args:
        num_simulations: Number of simulations (default 30)
        base_seed: Base random seed (default 42)
    """
    print("=" * 80)
    print("PREPARING ALL DATA SPLITS")
    print("=" * 80)
    print(f"\nGenerating {num_simulations} splits with seeds {base_seed} to {base_seed + num_simulations - 1}")
    print("This will take a few minutes but only needs to be done once.\n")
    
    # First split will download from Hugging Face
    print("Downloading dataset from Hugging Face (first time only)...")
    data_loader = DataLoader(
        dataset_name='amazon_reviews',
        random_state=base_seed
    )
    train_df, val_df, test_df = data_loader.load()
    data_loader.save_splits(format='csv')
    print(f"✓ Saved split for seed {base_seed}")
    
    # Remaining splits will reuse the downloaded data
    print(f"\nGenerating remaining {num_simulations - 1} splits...")
    for sim_id in tqdm(range(1, num_simulations), desc="Generating splits"):
        seed = base_seed + sim_id
        
        # Create new data loader with different seed
        data_loader = DataLoader(
            dataset_name='amazon_reviews',
            random_state=seed
        )
        
        # Load and save (will use cached Hugging Face data)
        train_df, val_df, test_df = data_loader.load()
        data_loader.save_splits(format='csv')
    
    print("\n" + "=" * 80)
    print("ALL SPLITS PREPARED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nSaved {num_simulations} splits to data/raw/")
    print("Future simulation runs will be much faster!")
    print("\nFiles created:")
    print(f"  - data/raw/train/train_seed{base_seed}.csv to train_seed{base_seed + num_simulations - 1}.csv")
    print(f"  - data/raw/validation/validation_seed{base_seed}.csv to validation_seed{base_seed + num_simulations - 1}.csv")
    print(f"  - data/raw/test/test_seed{base_seed}.csv to test_seed{base_seed + num_simulations - 1}.csv")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-generate all data splits for simulations')
    parser.add_argument('--num-simulations', type=int, default=30,
                       help='Number of simulations (default: 30)')
    parser.add_argument('--base-seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    
    args = parser.parse_args()
    
    prepare_all_splits(
        num_simulations=args.num_simulations,
        base_seed=args.base_seed
    )
