# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Google Colab notebook (`notebooks/sentiment_analysis_colab.ipynb`) for cloud-based execution
- Jupyter notebook infrastructure with template and creation scripts
- Notebook documentation in `notebooks/README.md`
- Interactive notebook support with VS Code cell markers
- Comprehensive documentation structure in `docs/` directory
- Professional folder organization for data pipeline
- Data structure with train/validation/test separation
- Seed-based file naming for reproducibility
- Support for multiple data formats (CSV, Parquet)
- Utility scripts for verification
- Contributing guidelines
- This changelog

### Changed
- Reorganized data folders into `raw/`, `processed/`, and `perturbed/` with subfolders
- Updated `DataLoader` class with new save/load methods
- Updated path references in analysis scripts
- Improved README with documentation links
- Moved all demo, validation, and verification scripts to `scripts/` folder
- Cleaned up root directory (removed duplicate files)

### Removed
- Duplicate `advanced_analysis.py` (kept in `src/`)
- Duplicate `statistical_tests.py` (kept as `src/statistical_validator.py`)

### Fixed
- Data leakage prevention through physical folder separation
- File path references in `advanced_analysis.py`

## [1.0.0] - 2026-02-04

### Added
- Initial project structure
- Data loading and preprocessing modules
- SVM classifiers (BoW and Embeddings)
- BERT classifier implementation
- Statistical validation framework
- Advanced NLP analysis tools
- Visualization module
- Comprehensive test suite
- Requirements specification

### Features
- Multiple simulation support (10-30 runs)
- Statistical significance testing
- Text length analysis
- Robustness to typos evaluation
- Emoji impact analysis
- Professional visualizations

---

## Version History

### Version Numbering

- **Major version** (X.0.0): Breaking changes, major features
- **Minor version** (0.X.0): New features, backward compatible
- **Patch version** (0.0.X): Bug fixes, minor improvements

### Release Process

1. Update version in relevant files
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push origin v1.0.0`

---

## Categories

### Added
New features or functionality

### Changed
Changes to existing functionality

### Deprecated
Features that will be removed in future versions

### Removed
Features that have been removed

### Fixed
Bug fixes

### Security
Security-related changes

---

**Note**: This changelog follows the principles of [Keep a Changelog](https://keepachangelog.com/).
