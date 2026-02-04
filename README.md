# Sentiment Analysis on Product Reviews

A comprehensive Natural Language Processing (NLP) project comparing classical and modern approaches for sentiment classification on product reviews.

## Project Overview

This project implements and compares four sentiment classification approaches:
- **SVM + Bag of Words (TF-IDF)**: Classical baseline with interpretable features
- **SVM + Word Embeddings**: Dense representations using pre-trained GloVe/Word2Vec
- **BERT (DistilBERT)**: Fine-tuned transformer model
- **In-Context Learning (Bonus)**: Few-shot learning with LLMs

The project emphasizes **scientific rigor** through:
- Multiple simulation runs (10-30) with different random seeds
- Statistical validation (Wilcoxon, Kruskal-Wallis tests)
- Advanced NLP analyses (text length, typos, emojis, sarcasm, formality)
- Professional visualizations for academic presentation

## Key Features

- **Data Leakage Prevention**: Strict train/validation/test separation (70/15/15)
- **Statistical Validation**: Significance testing with α=0.05 (95% confidence)
- **Advanced Analyses**:
  - Text length vs accuracy correlation
  - Robustness to typos and spelling errors
  - Impact of emojis on classification
  - Sarcasm and irony detection challenges
  - Sensitivity to formality and dialect
- **Professional Visualizations**: Boxplots, heatmaps, p-value matrices, confidence intervals

## Project Structure

```
sentiment-analysis-product-reviews/
├── data/                       # Data storage (organized by pipeline stage)
│   ├── raw/                    # Original splits (train/validation/test)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── processed/              # Preprocessed data (model-specific)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   └── perturbed/              # Perturbed test data for robustness
│       └── test/
├── docs/                       # Project documentation
│   ├── README.md               # Documentation index
│   ├── architecture/           # System design documents
│   │   ├── DATA_STRUCTURE.md
│   │   └── DATA_FLOW_DIAGRAM.md
│   ├── guides/                 # User guides and tutorials
│   │   ├── QUICK_START_DATA.md
│   │   ├── CHANGES_SUMMARY.md
│   │   └── VISUALIZER_QUICK_REFERENCE.md
│   └── examples/               # Code examples
│       └── example_data_structure.py
├── notebooks/                  # Jupyter notebooks (optional)
│   └── .gitkeep
├── src/                        # Source code modules
│   ├── config.py               # Experiment configuration
│   ├── data_loader.py          # Data loading and splitting
│   ├── preprocessor.py         # Text preprocessing
│   ├── vectorizers.py          # TF-IDF vectorization
│   ├── embedding_encoder.py    # Word embeddings
│   ├── classifiers.py          # SVM classifiers
│   ├── bert_classifier.py      # BERT fine-tuning
│   ├── evaluator.py            # Metrics calculation
│   ├── statistical_validator.py # Statistical tests
│   ├── simulation_runner.py    # Multiple simulation orchestration
│   ├── advanced_analysis.py    # Advanced NLP analyses
│   └── visualizer.py           # Visualization generation
├── scripts/                    # Utility scripts
│   ├── README.md               # Scripts documentation
│   ├── test_folder_structure.py # Verify folder structure
│   ├── demo_*.py               # Module demonstrations
│   ├── validate_*.py           # Pipeline validation
│   └── verify_*.py             # Dependency verification
├── results/                    # Experiment results
│   ├── simulations/            # Metrics from multiple runs
│   ├── statistical_tests/      # Statistical reports
│   ├── advanced_analysis/      # Advanced NLP analysis results
│   └── plots/                  # All visualizations
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── .kiro/                      # Kiro IDE configuration
│   └── specs/                  # Feature specifications
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended for BERT training (8GB VRAM minimum)
- API key for LLM (OpenAI, Claude, or Gemini) for bonus ICL classifier

### Installation

1. Clone the repository:
```bash
git clone https://github.com/EvelinLimeira/sentiment-analysis-product-reviews.git
cd sentiment-analysis-product-reviews
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Dataset

The project uses product review datasets with ratings (1-5 stars):
- **Positive**: 4-5 stars
- **Negative**: 1-2 stars
- **Neutral** (discarded): 3 stars

Recommended datasets:
- Amazon Product Reviews (Kaggle)
- Yelp Reviews
- IMDB Reviews (adapted)

Minimum: 3,000 reviews | Ideal: 5,000+ reviews

## Usage

### 1. Data Preparation

```python
from src.data_loader import DataLoader

# Initialize with seed for reproducibility
loader = DataLoader(dataset_name='amazon_reviews', random_state=42)

# Load and split data (70/15/15)
train_df, val_df, test_df = loader.load()

# Save splits to organized structure
loader.save_splits(format='csv')
# Creates: data/raw/train/train_seed42.csv, etc.
```

For more details, see [Data Structure Guide](docs/guides/QUICK_START_DATA.md).

### 2. Train Models

```python
from src.classifiers import SVMClassifier
from src.bert_classifier import BERTClassifier

# SVM + BoW
svm_bow = SVMClassifier(kernel='linear')
svm_bow.fit(X_train_tfidf, y_train)

# BERT
bert = BERTClassifier(model_name='distilbert-base-uncased')
bert.fit(train_texts, train_labels, val_texts, val_labels)
```

### 3. Statistical Validation

```python
from src.statistical_validator import StatisticalValidator

validator = StatisticalValidator(alpha=0.05)
validator.generate_report(
    model_names=['svm_bow', 'svm_embeddings', 'bert', 'llm'],
    metrics=['accuracy', 'f1_score']
)
```

### 4. Advanced Analysis

```python
from src.advanced_analysis import AdvancedNLPAnalysis

analyzer = AdvancedNLPAnalysis(test_df, predictions_dict)
analyzer.analyze_length_vs_accuracy()
analyzer.analyze_emoji_impact()
analyzer.analyze_formality()
```

## Results

Results include:
- **Comparison tables** with mean ± std across simulations
- **Statistical significance** tests (p-values, winners)
- **Confusion matrices** for each model
- **Boxplots** showing metric distributions
- **Advanced analyses** revealing model behavior patterns

All visualizations are exported in high resolution (300 DPI) for presentations.

## Testing

Run unit tests:
```bash
pytest tests/unit/
```

Run integration tests:
```bash
pytest tests/integration/
```

Run all tests:
```bash
pytest
```

Verify folder structure:
```bash
python scripts/test_folder_structure.py
```

## Dependencies

Key libraries:
- **NLP**: scikit-learn, nltk, gensim, transformers, torch
- **Statistics**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, hypothesis
- **Utilities**: pandas, numpy, tqdm, emoji

See `requirements.txt` for complete list with versions.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Quick Start Guide](docs/guides/quick-start.md)** - Get started quickly
- **[Data Structure](docs/architecture/data-structure.md)** - Folder organization
- **[Data Flow](docs/architecture/data-flow.md)** - Pipeline visualization
- **[Visualizer Guide](docs/guides/visualizer.md)** - Create visualizations
- **[Examples](docs/examples/)** - Working code examples

See [docs/README.md](docs/README.md) for complete documentation index.

## Contributing

See [contributing.md](contributing.md) for guidelines on:
- Code standards and style
- Testing requirements
- Commit message format
- Pull request process

## Changelog

See [changelog.md](changelog.md) for version history and changes.

This is an academic project for a Natural Language Processing course. See [contributing.md](contributing.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Evelin Limeira**
- GitHub: [@EvelinLimeira](https://github.com/EvelinLimeira)

## Acknowledgments

- Course: Natural Language Processing
- Emphasis on combining **simplicity** with **scientific rigor**
- Inspired by classical and modern NLP research

## Contact

For questions or collaboration opportunities, please open an issue or contact via GitHub.

---

**Note**: This project prioritizes reproducibility and statistical validity. All experiments use fixed random seeds and multiple simulation runs to ensure robust conclusions.
