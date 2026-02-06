# Sentiment Analysis on Product Reviews

A comprehensive Natural Language Processing (NLP) project comparing classical and modern approaches for sentiment classification on product reviews.

## Project Overview

This project implements and compares four sentiment classification approaches:
- **SVM + Bag of Words (TF-IDF)**: Classical baseline with interpretable features
- **SVM + Word Embeddings**: Dense representations using pre-trained GloVe/Word2Vec
- **BERT (DistilBERT)**: Fine-tuned transformer model
- **In-Context Learning (Bonus)**: Few-shot learning with LLMs

The project emphasizes **scientific rigor** through:
- 10 simulation runs with different random seeds (42-51)
- Statistical validation (Wilcoxon, Kruskal-Wallis tests)
- Advanced NLP analyses (text length, typos, emojis, sarcasm, formality)
- Professional visualizations for academic presentation
- BERT optimization: 10 epochs, batch size 32, early stopping (patience=3)

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

### Quick Start Options

#### Option 1: Google Colab (Recommended for Beginners)
Run the project in your browser with free GPU access:
1. Open [sentiment_analysis_colab.ipynb](notebooks/sentiment_analysis_colab.ipynb) in Google Colab
2. Select GPU runtime: `Runtime` → `Change runtime type` → `GPU`
3. Run all cells
4. **Try the interactive demo** in Section 11 to test models with your own reviews!

**Troubleshooting:** If you encounter issues with results not being saved or other Colab-specific problems, see the [Colab Troubleshooting Guide](docs/guides/colab-troubleshooting.md).

#### Option 2: Local Installation

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

#### Option 3: Jupyter Notebook
See [notebooks/README.md](notebooks/README.md) for interactive notebook options.

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

### Interactive Notebooks

For a guided walkthrough, use the Jupyter notebooks:

- **[Google Colab](notebooks/sentiment_analysis_colab.ipynb)**: Run in browser with free GPU
- **[Local Notebook](notebooks/sentiment_analysis_pipeline.ipynb)**: Run locally with Jupyter
- See [notebooks/README.md](notebooks/README.md) for more options

### Command Line Usage

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
- **[BERT Training Guide](docs/guides/bert-training.md)** - BERT configuration and optimization
- **[Interactive Demo Guide](docs/guides/interactive-demo.md)** - Test models with your own reviews
- **[Colab Troubleshooting](docs/guides/colab-troubleshooting.md)** - Fix common Colab issues
- **[Data Structure](docs/architecture/data-structure.md)** - Folder organization
- **[Data Flow](docs/architecture/data-flow.md)** - Pipeline visualization
- **[Visualizer Guide](docs/guides/visualizer.md)** - Create visualizations
- **[Examples](docs/examples/)** - Working code examples

See [docs/README.md](docs/README.md) for complete documentation index.

---

## 📊 Experiment Results

### Executive Summary

We conducted a comparison of three sentiment classification approaches using **10 independent simulations** with different data splits (seeds 42-51) to ensure statistical validity. All results are statistically significant (p < 0.05).

### Performance Comparison

| Model | Accuracy | F1-Score | Precision | Recall | Train Time |
|-------|----------|----------|-----------|--------|------------|
| **BERT (DistilBERT)** | **91.58% ± 0.80%** | **91.58% ± 0.80%** | **91.61% ± 0.79%** | **91.58% ± 0.79%** | ~8.6 min |
| SVM + Bag of Words | 85.90% ± 1.45% | 84.38% ± 2.29% | 82.42% ± 1.51% | 85.56% ± 1.63% | ~1.6 min |
| SVM + Embeddings | 79.97% ± 1.76% | 79.70% ± 1.31% | 78.90% ± 2.71% | 81.20% ± 1.48% | ~2.0 min |

**Key Findings:**
- BERT achieves **+7.20% F1-Score improvement** over SVM+BoW
- All differences are **statistically significant** (Wilcoxon test, p < 0.002)
- BERT shows **lower variance**, indicating more stable performance
- 95% Confidence Interval for BERT F1-Score: **[91.28%, 91.87%]**

### Visualizations

#### Model Performance Comparison
![Metrics Comparison](results/plots/metrics_comparison.png)
*Comparison of accuracy and F1-score across all models*

#### Radar Plot - Overall Performance
![Radar Comparison](results/plots/radar_comparison.png)
*Multi-dimensional comparison across Accuracy, Precision, Recall, and F1-Score*

#### Statistical Distribution
![Boxplots Comparison](results/plots/boxplots_comparison.png)
*Distribution of accuracy and F1-score across 10 simulations*

#### Confidence Intervals
![Confidence Intervals](results/plots/confidence_intervals_f1_macro.png)
*95% confidence intervals showing non-overlapping ranges*

#### Statistical Significance
![P-value Matrix](results/plots/pvalue_matrix_f1_macro.png)
*Pairwise statistical tests (Wilcoxon) - all p-values < 0.05*

### Advanced Analysis Results

#### Robustness to Typos
| Model | Original Accuracy | With 5% Typos | Degradation |
|-------|-------------------|---------------|-------------|
| **BERT** | 90.33% | 89.93% | **0.44%** |
| SVM+Embeddings | 76.53% | 75.93% | 0.78% |
| SVM+BoW | 79.27% | 79.40% | -0.17% |

![Typo Robustness](results/plots/advanced_analysis/typo_robustness.png)

#### Sarcasm Detection
| Model | Sarcasm Accuracy | Normal Accuracy | Degradation |
|-------|------------------|-----------------|-------------|
| **BERT** | **88.00%** | 90.41% | **2.67%** |
| SVM+Embeddings | 60.00% | 77.10% | 22.18% |
| SVM+BoW | 56.00% | 80.07% | 30.06% |

![Sarcasm Analysis](results/plots/advanced_analysis/sarcasm_analysis.png)

#### Formality Sensitivity
| Model | Formal | Informal | Excited |
|-------|--------|----------|---------|
| **BERT** | **90.35%** | **88.04%** | **94.12%** |
| SVM+BoW | 78.33% | 86.96% | 90.20% |
| SVM+Embeddings | 76.20% | 78.26% | 82.35% |

![Formality Analysis](results/plots/advanced_analysis/formality_heatmap.png)

#### Radar Plot - Robustness Analysis
![Radar Advanced](results/plots/radar_advanced_analysis.png)
*Model robustness across different text characteristics (normal, typos, sarcasm, formality)*

### Methodology

- **10 simulations** with different random seeds (42-51) for statistical validity
- Each simulation uses a **different data split** to test generalization
- **Statistical tests:** Wilcoxon (pairwise), Kruskal-Wallis (multiple groups)
- **Significance level:** α = 0.05 (95% confidence)
- **BERT configuration:** 10 epochs, batch size 32, early stopping (patience=3)

### Complete Report

For detailed methodology, statistical analysis, and discussion, see the complete [Experiment Report](EXPERIMENT_REPORT.md).

---

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

**Bruno Lutiano**

## Acknowledgments

- Course: Natural Language Processing
- Emphasis on combining **simplicity** with **scientific rigor**
- Inspired by classical and modern NLP research

## Contact

For questions or collaboration opportunities, please open an issue or contact via GitHub.

---

**Note**: This project prioritizes reproducibility and statistical validity. All experiments use fixed random seeds and multiple simulation runs to ensure robust conclusions.
