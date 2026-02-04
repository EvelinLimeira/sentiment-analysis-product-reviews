# Utility Scripts

Collection of utility scripts for project maintenance, verification, and demonstration.

## Available Scripts

### Verification Scripts

#### test_folder_structure.py
Verifies that the data folder structure is correctly set up.

**Usage:**
```bash
python scripts/test_folder_structure.py
```

**When to Use:**
- After cloning the repository
- Before running experiments
- When troubleshooting data loading issues

#### verify_bert_requirements.py
Verifies BERT dependencies and GPU availability.

**Usage:**
```bash
python scripts/verify_bert_requirements.py
```

#### verify_visualizer_requirements.py
Verifies visualization dependencies.

**Usage:**
```bash
python scripts/verify_visualizer_requirements.py
```

### Validation Scripts

#### validate_all_classifiers.py
Trains and evaluates all classifiers (SVM+BoW, SVM+Embeddings, BERT) for complete pipeline validation.

**Usage:**
```bash
python scripts/validate_all_classifiers.py
```

**Note:** This runs full training and may take time.

#### validate_all_classifiers_quick.py
Quick validation with reduced parameters for faster testing.

**Usage:**
```bash
python scripts/validate_all_classifiers_quick.py
```

### Demo Scripts

#### demo_evaluator.py
Demonstrates the Evaluator module functionality.

**Usage:**
```bash
python scripts/demo_evaluator.py
```

#### demo_simulation_runner.py
Demonstrates running multiple simulations for statistical validation.

**Usage:**
```bash
python scripts/demo_simulation_runner.py
```

#### demo_statistical_validator.py
Demonstrates statistical significance testing.

**Usage:**
```bash
python scripts/demo_statistical_validator.py
```

#### demo_visualizer_integration.py
Demonstrates visualization generation.

**Usage:**
```bash
python scripts/demo_visualizer_integration.py
```

## Script Categories

### Verification (3 scripts)
Quick checks to verify setup and dependencies.

### Validation (2 scripts)
End-to-end pipeline validation with model training.

### Demo (4 scripts)
Demonstrations of individual module functionality.

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
- Prefix with action: `test_`, `validate_`, `demo_`, `verify_`

## Best Practices

- **Idempotent**: Scripts should be safe to run multiple times
- **Informative**: Provide clear output about what's happening
- **Fail Fast**: Exit early with clear error messages
- **Documented**: Include usage examples and expected behavior
- **Tested**: Test scripts with various scenarios

---

**Last Updated**: February 4, 2026
