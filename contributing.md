# Contributing Guidelines

Thank you for your interest in contributing to the Sentiment Analysis NLP project!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Documentation Standards](#documentation-standards)
- [Testing Requirements](#testing-requirements)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This is an academic project. We expect:
- Professional and respectful communication
- Constructive feedback
- Focus on code quality and scientific rigor
- Proper attribution of sources

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/EvelinLimeira/sentiment-analysis-product-reviews.git
   cd sentiment-analysis-product-reviews
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Verify setup**
   ```bash
   python scripts/test_folder_structure.py
   pytest tests/
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**
   ```bash
   pytest tests/
   python scripts/test_folder_structure.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Standards

### Python Style

- **PEP 8**: Follow Python style guide
- **Type Hints**: Use type annotations for function signatures
- **Docstrings**: Use Google-style docstrings
- **Line Length**: Maximum 100 characters
- **Imports**: Group and sort (stdlib, third-party, local)

**Example:**
```python
from typing import List, Tuple
import pandas as pd
from src.config import ExperimentConfig


def load_data(filepath: str, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split dataset.
    
    Args:
        filepath: Path to the dataset file
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If filepath doesn't exist
    """
    # Implementation
    pass
```

### Code Organization

- **Modules**: One class per file (unless tightly coupled)
- **Functions**: Keep functions small and focused (< 50 lines)
- **Classes**: Use classes for stateful operations
- **Constants**: UPPER_CASE for module-level constants

### Error Handling

- Use specific exceptions
- Provide informative error messages
- Log errors appropriately
- Clean up resources (use context managers)

**Example:**
```python
class DatasetNotFoundError(Exception):
    """Raised when dataset cannot be loaded."""
    pass


def load_dataset(name: str) -> pd.DataFrame:
    """Load dataset by name."""
    try:
        return pd.read_csv(f'data/raw/{name}.csv')
    except FileNotFoundError:
        raise DatasetNotFoundError(
            f"Dataset '{name}' not found in data/raw/. "
            f"Available datasets: {list_available_datasets()}"
        )
```

## Documentation Standards

### Code Documentation

- **Module docstrings**: Describe module purpose and requirements
- **Class docstrings**: Describe class purpose and attributes
- **Function docstrings**: Describe parameters, returns, raises
- **Inline comments**: Explain complex logic (not obvious code)

### Project Documentation

Documentation lives in `docs/`:

- **Architecture docs** → `docs/architecture/`
- **User guides** → `docs/guides/`
- **Code examples** → `docs/examples/`

### Documentation Format

Use Markdown with:
- Clear headings hierarchy
- Code blocks with language tags
- Tables for structured data
- Links to related documents

**Example:**
```markdown
# Feature Name

Brief description of the feature.

## Usage

\`\`\`python
from src.module import Feature

feature = Feature()
result = feature.process()
\`\`\`

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Feature name |
| value | int | Feature value |

## See Also

- [Related Guide](guides/related.md)
- [API Reference](architecture/api.md)
```

## Testing Requirements

### Test Coverage

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **Minimum coverage**: 80% for new code

### Test Organization

```
tests/
├── unit/                  # Unit tests
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── ...
└── integration/           # Integration tests
    ├── test_data_pipeline.py
    └── ...
```

### Writing Tests

Use pytest with descriptive names:

```python
def test_data_loader_splits_data_correctly():
    """Test that DataLoader creates correct train/val/test splits."""
    loader = DataLoader(test_size=0.2, val_size=0.1, random_state=42)
    train, val, test = loader.load()
    
    total = len(train) + len(val) + len(test)
    assert len(train) / total == pytest.approx(0.7, abs=0.01)
    assert len(val) / total == pytest.approx(0.1, abs=0.01)
    assert len(test) / total == pytest.approx(0.2, abs=0.01)
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_data_loader.py

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(data_loader): add support for parquet format

- Add load_parquet() method
- Update save_splits() to support parquet
- Add tests for parquet functionality

Closes #123
```

```
fix(preprocessor): handle empty strings correctly

Previously crashed on empty strings. Now returns empty list.

Fixes #456
```

```
docs(architecture): update data structure diagram

Add perturbed data folder to diagram
```

## Pull Request Process

1. **Update documentation**
   - Update relevant docs in `docs/`
   - Update README if needed
   - Add docstrings to new code

2. **Add tests**
   - Write unit tests for new functions
   - Write integration tests for new features
   - Ensure all tests pass

3. **Update CHANGELOG** (if exists)
   - Add entry under "Unreleased"
   - Describe changes clearly

4. **Create PR**
   - Use descriptive title
   - Reference related issues
   - Describe changes and rationale
   - Add screenshots if UI changes

5. **Code Review**
   - Address reviewer feedback
   - Keep discussion professional
   - Update PR as needed

6. **Merge**
   - Squash commits if many small commits
   - Use merge commit for feature branches
   - Delete branch after merge

## Project-Specific Guidelines

### Data Handling

- **Never commit data files** (use .gitignore)
- **Use seed-based filenames** for reproducibility
- **Follow folder structure** (raw/processed/perturbed)
- **Document data sources** and preprocessing steps

### Model Development

- **Save model checkpoints** with descriptive names
- **Log hyperparameters** for reproducibility
- **Track experiments** (consider MLflow/Weights & Biases)
- **Document model architecture** and training process

### Statistical Validation

- **Use fixed random seeds** for reproducibility
- **Run multiple simulations** (minimum 10)
- **Report confidence intervals** with results
- **Use appropriate statistical tests** (document choice)

## Questions?

- Check existing [documentation](docs/README.md)
- Review [examples](docs/examples/)
- Open an issue for discussion

---

**Thank you for contributing!**
