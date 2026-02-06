# Notebooks

This directory contains Jupyter notebooks for the sentiment analysis pipeline.

## Main Notebook

### `notebook_template.py`
Python script with notebook cells (use `# %%` markers). Can be:
- Opened directly in VS Code as an interactive notebook
- Converted to `.ipynb` using `jupytext` or similar tools
- Run as a regular Python script

**Sections:**
1. Imports and Configuration
2. Data Loading and Exploration
3. Text Preprocessing
4. SVM + Bag of Words
5. SVM + Embeddings
6. BERT Classifier (10 epochs, batch 32, early stopping)
7. Comparison and Visualizations
8. Advanced NLP Analysis
9. Conclusions

**Key Features:**
- 10 simulations with different data splits (seeds 42-51)
- BERT: 10 epochs, batch size 32, early stopping (patience=3)
- Statistical validation with Wilcoxon and Kruskal-Wallis tests
- Advanced analyses: typos, sarcasm, formality, text length

## Usage

### Option 1: VS Code (Recommended)
1. Open `notebook_template.py` in VS Code
2. VS Code will recognize `# %%` markers as cells
3. Run cells interactively using the Python extension

### Option 2: Convert to Jupyter Notebook
```bash
# Install jupytext
pip install jupytext

# Convert to notebook
jupytext --to notebook notebook_template.py

# Or use the create script
python create_notebook.py
```

### Option 3: Run as Script
```bash
python notebook_template.py
```

## Creating Custom Notebooks

Use `create_notebook.py` to programmatically generate notebooks with custom sections.

## Requirements

All required packages are listed in `requirements.txt` at the project root.

## Notes

- Notebooks use relative imports (`sys.path.append('..')`)
- Run from the `notebooks/` directory
- BERT training may take 30-60 minutes depending on hardware
- Results are saved to `../results/` directory
