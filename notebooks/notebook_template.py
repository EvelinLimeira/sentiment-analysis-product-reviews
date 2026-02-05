# %% [markdown]
# # Sentiment Analysis Pipeline for Product Reviews
# 
# ## Course: Natural Language Processing
# ## Date: 2024
# 
# ---
# 
# ## Overview
# 
# This notebook implements and compares sentiment classification approaches:
# 1. **SVM + Bag of Words (TF-IDF)**
# 2. **SVM + Word Embeddings (GloVe)**
# 3. **BERT (DistilBERT)**
# 
# The analysis includes:
# - Multiple simulation runs for statistical validation
# - Statistical significance testing
# - Advanced NLP analyses
# - Professional visualizations

# %% [markdown]
# ## Section 1: Imports and Configuration

# %%
# Standard libraries
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('..')

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Our modules
from src.config import ExperimentConfig
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.vectorizers import BoWVectorizer
from src.embedding_encoder import EmbeddingEncoder
from src.classifiers import SVMClassifier
from src.bert_classifier import BERTClassifier
from src.evaluator import Evaluator
from src.statistical_validator import StatisticalValidator
from src.advanced_analysis import AdvancedNLPAnalysis
from src.visualizer import Visualizer
from src.simulation_runner import SimulationRunner

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("✓ All libraries imported successfully")

# %%
# Configure experiment
config = ExperimentConfig(
    dataset_name='amazon_reviews',
    num_simulations=10
)

print("Experiment Configuration:")
print(config)

# %% [markdown]
# ## Section 2: Data Loading and Exploration

# %%
# Load data
data_loader = DataLoader(random_state=RANDOM_SEED)
train_df, val_df, test_df = data_loader.load()

print(f"Dataset loaded!")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# %%
# Class distribution
distribution = data_loader.get_class_distribution()
print("\\nClass Distribution:")
for split, counts in distribution.items():
    total = counts['negative'] + counts['positive']
    print(f"  {split}: Neg={counts['negative']} ({counts['negative']/total*100:.1f}%), "
          f"Pos={counts['positive']} ({counts['positive']/total*100:.1f}%)")

# %%
# Sample reviews
print("Sample Reviews:")
for i, row in train_df.head(3).iterrows():
    label = "POSITIVE" if row['label'] == 1 else "NEGATIVE"
    print(f"\\n[{label}] {row['text'][:150]}...")

# %% [markdown]
# ## Section 3: Text Preprocessing

# %%
# Create and fit preprocessor
preprocessor = Preprocessor(language='english', remove_stopwords=True)
train_texts = train_df['text'].tolist()
preprocessor.fit(train_texts)

print(f"Preprocessor fitted")
print(f"Vocabulary size: {preprocessor.get_vocabulary_size()}")

# %% [markdown]
# ## Section 4: SVM + Bag of Words

# %%
import time

# Preprocess
train_texts_processed = preprocessor.transform(train_texts)
test_texts_processed = preprocessor.transform(test_df['text'].tolist())

# Vectorize
vectorizer = BoWVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts_processed)
X_test = vectorizer.transform(test_texts_processed)

# Train
start = time.time()
classifier_bow = SVMClassifier(kernel='linear', C=1.0)
classifier_bow.fit(X_train, train_df['label'].values)
train_time = time.time() - start

# Predict
start = time.time()
preds_bow = classifier_bow.predict(X_test)
infer_time = time.time() - start

# Evaluate
evaluator = Evaluator()
metrics = evaluator.evaluate(test_df['label'].values, preds_bow, 'svm_bow')

print(f"\\nSVM + BoW Results:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  F1-Score: {metrics['f1_macro']:.4f}")
print(f"  Training: {train_time:.2f}s")
print(f"  Inference: {infer_time:.2f}s")

# %% [markdown]
# ## Section 5: SVM + Embeddings

# %%
# Encode with embeddings
encoder = EmbeddingEncoder(model_name='glove-wiki-gigaword-100')
X_train_emb = encoder.encode_batch(train_texts_processed)
X_test_emb = encoder.encode_batch(test_texts_processed)

# Train
start = time.time()
classifier_emb = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
classifier_emb.fit(X_train_emb, train_df['label'].values)
train_time = time.time() - start

# Predict
start = time.time()
preds_emb = classifier_emb.predict(X_test_emb)
infer_time = time.time() - start

# Evaluate
metrics = evaluator.evaluate(test_df['label'].values, preds_emb, 'svm_embeddings')

print(f"\\nSVM + Embeddings Results:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  F1-Score: {metrics['f1_macro']:.4f}")

# %% [markdown]
# ## Section 6: BERT Classifier

# %%
# Train BERT (this takes longer)
print("Training BERT...")
classifier_bert = BERTClassifier(
    model_name='distilbert-base-uncased',
    batch_size=16,
    num_epochs=3
)

start = time.time()
classifier_bert.fit(
    train_df['text'].tolist(), train_df['label'].tolist(),
    val_df['text'].tolist(), val_df['label'].tolist()
)
train_time = time.time() - start

# Predict
start = time.time()
preds_bert = classifier_bert.predict(test_df['text'].tolist())
infer_time = time.time() - start

# Evaluate
metrics = evaluator.evaluate(test_df['label'].values, preds_bert, 'bert')

print(f"\\nBERT Results:")
print(f"  Accuracy: {metrics['accuracy']:.4f}")
print(f"  F1-Score: {metrics['f1_macro']:.4f}")

# %% [markdown]
# ## Section 7: Comparison and Visualizations

# %%
# Get comparison table
comparison = evaluator.get_comparison_table()
print("\\nModel Comparison:")
print(comparison)

# %%
# Visualize results
viz = Visualizer()

# Metrics comparison
viz.plot_metrics_comparison(
    evaluator.results,
    metrics=['accuracy', 'f1_macro']
)
plt.show()

# Confusion matrices
for model_name, cm in evaluator.confusion_matrices.items():
    viz.plot_confusion_matrix(cm, model_name)
    plt.show()

# %% [markdown]
# ## Section 8: Advanced NLP Analysis

# %%
# Prepare predictions dictionary
predictions = {
    'svm_bow': preds_bow,
    'svm_embeddings': preds_emb,
    'bert': preds_bert
}

# Create analyzer
analyzer = AdvancedNLPAnalysis(test_df, predictions)

# Run analyses
results = analyzer.run_all_analyses()

# %% [markdown]
# ## Section 9: Conclusions
# 
# ### Key Findings:
# 
# 1. **Model Performance**: [Add your analysis]
# 2. **Statistical Significance**: [Add your analysis]
# 3. **Advanced Analyses**: [Add your analysis]
# 
# ### Recommendations:
# 
# [Add your recommendations]
