# Utility Scripts

Collection of utility scripts for project maintenance and verification.

## Available Scripts

### test_folder_structure.py

Verifies that the data folder structure is correctly set up.

**Usage:**
```bash
python scripts/test_folder_structure.py
```

**Expected Output:**
```
Checking folder structure...
============================================================
✓ data\raw\train
✓ data\raw\validation
✓ data\raw\test
✓ data\processed\train
✓ data\processed\validation
✓ data\processed\test
✓ data\perturbed\test
============================================================
✓ All folders exist!
```

**When to Use:**
- After cloning the repository
- Before running experiments
- After making changes to data structure
- When troubleshooting data loading issues

**Exit Codes:**
- `0`: All folders exist (success)
- `1`: Some folders are missing (failure)

## Adding New Scripts

When adding utility scripts:

1. Place them in the `scripts/` directory
2. Add a descriptive docstring at the top
3. Make them executable with proper error handling
4. Document them in this README
5. Use clear exit codes (0 for success, non-zero for errors)

## Script Naming Convention

Follow these conventions:
- Use snake_case: `test_folder_structure.py`
- Be descriptive: `validate_data_splits.py` not `check.py`
- Prefix with action: `test_`, `validate_`, `generate_`, `cleanup_`

## Best Practices

- **Idempotent**: Scripts should be safe to run multiple times
- **Informative**: Provide clear output about what's happening
- **Fail Fast**: Exit early with clear error messages
- **Documented**: Include usage examples and expected behavior
- **Tested**: Test scripts with various scenarios

---

**Last Updated**: February 4, 2026
