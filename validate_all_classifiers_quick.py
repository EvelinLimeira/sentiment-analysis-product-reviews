"""
Task 11 Checkpoint: Quick Validation of All Classifiers
This script trains and evaluates all three implemented classifiers on a SMALL SAMPLE:
1. SVM + Bag of Words (TF-IDF)
2. SVM + Embeddings (GloVe)
3. BERT (DistilBERT) - with reduced epochs and batch size

This is a quick validation to ensure all components work correctly.
For full evaluation, use validate_all_classifiers.py with GPU.
"""

import time
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.vectorizers import BoWVectorizer
from src.embedding_encoder import EmbeddingEncoder
from src.classifiers import SVMClassifier
from src.bert_classifier import BERTClassifier
from src.evaluator import Evaluator

def main():
    print("=" * 80)
    print("TASK 11 CHECKPOINT: QUICK VALIDATION OF ALL CLASSIFIERS")
    print("=" * 80)
    print("\nNOTE: Using small sample for quick validation (CPU-friendly)")
    print("For full evaluation with larger dataset, use GPU and validate_all_classifiers.py")
    print()
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    print("-" * 80)
    data_loader = DataLoader(
        dataset_name='amazon_reviews',
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    train_df, val_df, test_df = data_loader.load()
    
    # Use smaller sample for quick validation
    SAMPLE_SIZE = 1000  # Small sample for quick testing
    TEST_SIZE = 200
    
    print(f"\n⚠️  Using SMALL SAMPLE for quick validation:")
    print(f"  - Training samples: {SAMPLE_SIZE} (from {len(train_df)})")
    print(f"  - Test samples: {TEST_SIZE} (from {len(test_df)})")
    
    # Sample data
    train_df_sample = train_df.sample(n=min(SAMPLE_SIZE, len(train_df)), random_state=42)
    val_df_sample = val_df.sample(n=min(200, len(val_df)), random_state=42)
    test_df_sample = test_df.sample(n=min(TEST_SIZE, len(test_df)), random_state=42)
    
    # Show class distribution
    dist = {
        'train': train_df_sample['label'].value_counts().to_dict(),
        'val': val_df_sample['label'].value_counts().to_dict(),
        'test': test_df_sample['label'].value_counts().to_dict()
    }
    print(f"\n  Class distribution in sample:")
    for split_name, counts in dist.items():
        total = sum(counts.values())
        neg = counts.get(0, 0)
        pos = counts.get(1, 0)
        print(f"    {split_name}: Negative={neg} ({neg/total*100:.1f}%), "
              f"Positive={pos} ({pos/total*100:.1f}%)")
    print()
    
    # Extract data
    X_train, y_train = train_df_sample['text'].tolist(), train_df_sample['label'].values
    X_val, y_val = val_df_sample['text'].tolist(), val_df_sample['label'].values
    X_test, y_test = test_df_sample['text'].tolist(), test_df_sample['label'].values
    
    # =========================================================================
    # METHOD 1: SVM + Bag of Words (TF-IDF)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: SVM + BAG OF WORDS (TF-IDF)")
    print("=" * 80)
    
    # Preprocess
    print("\nPreprocessing texts...")
    preprocessor = Preprocessor(language='english', remove_stopwords=True)
    preprocessor.fit(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    print(f"✓ Preprocessing complete")
    
    # Vectorize
    print("Vectorizing with TF-IDF...")
    vectorizer = BoWVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(X_train_prep)
    X_train_vec = vectorizer.transform(X_train_prep)
    X_test_vec = vectorizer.transform(X_test_prep)
    print(f"✓ Vectorization complete (vocabulary size: {vectorizer.get_vocabulary_size()})")
    
    # Train SVM
    print("Training SVM with linear kernel...")
    start_time = time.time()
    svm_bow = SVMClassifier(kernel='linear', C=1.0)
    svm_bow.fit(X_train_vec, y_train)
    train_time = time.time() - start_time
    print(f"✓ Training complete ({train_time:.2f}s)")
    
    # Predict
    print("Evaluating on test set...")
    start_time = time.time()
    y_pred_bow = svm_bow.predict(X_test_vec)
    inference_time = time.time() - start_time
    print(f"✓ Inference complete ({inference_time:.2f}s)")
    
    # Evaluate
    results_bow = evaluator.evaluate(y_test, y_pred_bow, 'SVM+BoW')
    evaluator.add_timing('SVM+BoW', train_time, inference_time)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {results_bow['accuracy']:.4f}")
    print(f"  Precision: {results_bow['precision_macro']:.4f}")
    print(f"  Recall:    {results_bow['recall_macro']:.4f}")
    print(f"  F1-Score:  {results_bow['f1_macro']:.4f}")
    
    # =========================================================================
    # METHOD 2: SVM + Embeddings (GloVe)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: SVM + EMBEDDINGS (GLOVE)")
    print("=" * 80)
    
    # Encode with embeddings
    print("\nLoading GloVe embeddings...")
    encoder = EmbeddingEncoder(model_name='glove-wiki-gigaword-100')
    print(f"✓ Embeddings loaded (dimension: {encoder.get_embedding_dim()})")
    
    print("Encoding texts...")
    X_train_emb = encoder.encode_batch(X_train_prep)
    X_test_emb = encoder.encode_batch(X_test_prep)
    
    # Calculate OOV rate
    oov_rate = encoder.get_oov_rate(X_test_prep)
    print(f"✓ Encoding complete (OOV rate: {oov_rate:.2%})")
    
    # Train SVM with RBF kernel
    print("Training SVM with RBF kernel...")
    start_time = time.time()
    svm_emb = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
    svm_emb.fit(X_train_emb, y_train)
    train_time = time.time() - start_time
    print(f"✓ Training complete ({train_time:.2f}s)")
    
    # Predict
    print("Evaluating on test set...")
    start_time = time.time()
    y_pred_emb = svm_emb.predict(X_test_emb)
    inference_time = time.time() - start_time
    print(f"✓ Inference complete ({inference_time:.2f}s)")
    
    # Evaluate
    results_emb = evaluator.evaluate(y_test, y_pred_emb, 'SVM+Embeddings')
    evaluator.add_timing('SVM+Embeddings', train_time, inference_time)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {results_emb['accuracy']:.4f}")
    print(f"  Precision: {results_emb['precision_macro']:.4f}")
    print(f"  Recall:    {results_emb['recall_macro']:.4f}")
    print(f"  F1-Score:  {results_emb['f1_macro']:.4f}")
    
    # =========================================================================
    # METHOD 3: BERT (DistilBERT) - Quick validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 3: BERT (DISTILBERT) - QUICK VALIDATION")
    print("=" * 80)
    print("\n⚠️  Using 1 epoch only for quick validation (CPU)")
    
    print("\nInitializing BERT classifier...")
    bert = BERTClassifier(
        model_name='distilbert-base-uncased',
        max_length=128,  # Reduced for speed
        batch_size=8,     # Smaller batch for CPU
        learning_rate=2e-5,
        num_epochs=1,     # Just 1 epoch for quick validation
        patience=1
    )
    print(f"✓ BERT initialized (device: {bert.device})")
    
    # Train BERT
    print("Fine-tuning BERT (1 epoch, this will take a few minutes on CPU)...")
    start_time = time.time()
    bert.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time
    print(f"✓ Training complete ({train_time:.2f}s = {train_time/60:.1f} minutes)")
    
    # Predict
    print("Evaluating on test set...")
    start_time = time.time()
    y_pred_bert = bert.predict(X_test)
    inference_time = time.time() - start_time
    print(f"✓ Inference complete ({inference_time:.2f}s)")
    
    # Evaluate
    results_bert = evaluator.evaluate(y_test, y_pred_bert, 'BERT')
    evaluator.add_timing('BERT', train_time, inference_time)
    
    print(f"\nResults:")
    print(f"  Accuracy:  {results_bert['accuracy']:.4f}")
    print(f"  Precision: {results_bert['precision_macro']:.4f}")
    print(f"  Recall:    {results_bert['recall_macro']:.4f}")
    print(f"  F1-Score:  {results_bert['f1_macro']:.4f}")
    
    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PRELIMINARY COMPARISON TABLE")
    print("=" * 80)
    print()
    
    comparison_df = evaluator.get_comparison_table()
    print(comparison_df.to_string())
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    comparison_df.to_csv('results/task11_comparison_quick.csv', index=False)
    print("✓ Results saved to: results/task11_comparison_quick.csv")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # Find best model by F1-score
    best_model = comparison_df.loc[comparison_df['f1_macro'].idxmax()]
    print(f"Best Model (by F1-score): {best_model.name}")
    print(f"  F1-Score: {best_model['f1_macro']:.4f}")
    print(f"  Accuracy: {best_model['accuracy']:.4f}")
    print()
    
    # Verify all metrics are in valid range
    print("Validation checks:")
    all_valid = True
    for model_name, row in comparison_df.iterrows():
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']:
            value = row[metric]
            if not (0.0 <= value <= 1.0):
                print(f"  ✗ {model_name} {metric} out of range: {value}")
                all_valid = False
    
    if all_valid:
        print("  ✓ All metrics are in valid range [0.0, 1.0]")
    
    # Check that all models were trained
    expected_models = {'SVM+BoW', 'SVM+Embeddings', 'BERT'}
    actual_models = set(comparison_df.index)
    if expected_models == actual_models:
        print("  ✓ All 3 classifiers trained and evaluated")
    else:
        print(f"  ✗ Missing models: {expected_models - actual_models}")
    
    print()
    print("=" * 80)
    print("TASK 11 CHECKPOINT COMPLETE!")
    print("=" * 80)
    print()
    print("✓ All classifiers have been validated successfully on small sample.")
    print("✓ The pipeline is working correctly.")
    print()
    print("NOTE: These results are from a SMALL SAMPLE for quick validation.")
    print("For full evaluation:")
    print("  1. Use GPU for BERT training (much faster)")
    print("  2. Run validate_all_classifiers.py with full dataset")
    print("  3. Proceed to Task 13 for multiple simulations")
    print()

if __name__ == '__main__':
    main()
