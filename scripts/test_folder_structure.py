"""
Quick test to verify the folder structure is created correctly.
"""

from pathlib import Path

def test_folder_structure():
    """Test that all required folders exist."""
    
    base_dir = Path('data')
    
    required_folders = [
        base_dir / 'raw' / 'train',
        base_dir / 'raw' / 'validation',
        base_dir / 'raw' / 'test',
        base_dir / 'processed' / 'train',
        base_dir / 'processed' / 'validation',
        base_dir / 'processed' / 'test',
        base_dir / 'perturbed' / 'test',
    ]
    
    print("Checking folder structure...")
    print("=" * 60)
    
    all_exist = True
    for folder in required_folders:
        exists = folder.exists() and folder.is_dir()
        status = "✓" if exists else "✗"
        print(f"{status} {folder}")
        if not exists:
            all_exist = False
    
    print("=" * 60)
    
    if all_exist:
        print("✓ All folders exist!")
        print("\nFolder structure:")
        print("data/")
        print("├── raw/")
        print("│   ├── train/")
        print("│   ├── validation/")
        print("│   └── test/")
        print("├── processed/")
        print("│   ├── train/")
        print("│   ├── validation/")
        print("│   └── test/")
        print("└── perturbed/")
        print("    └── test/")
        return True
    else:
        print("✗ Some folders are missing!")
        return False

if __name__ == "__main__":
    success = test_folder_structure()
    exit(0 if success else 1)
