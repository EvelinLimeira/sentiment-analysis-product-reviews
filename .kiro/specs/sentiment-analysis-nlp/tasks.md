# Implementation Plan: Sentiment Analysis NLP

## Overview

This plan implements the sentiment analysis pipeline for product reviews, comparing SVM+BoW, SVM+Embeddings, BERT, and ICL. The code will be developed in Jupyter Notebook with auxiliary Python modules. The implementation includes multiple simulation runs with statistical validation.

## Tasks

- [x] 1. Project setup and base structure
  - [x] 1.1 Create directory structure and initial files
    - Create `src/` with Python modules
    - Create `notebooks/` for Jupyter notebooks
    - Create `results/` with subdirectories (simulations/, statistical_tests/, advanced_analysis/, plots/)
    - Create `data/` with subdirectories (raw/, processed/, perturbed/)
    - Create `requirements.txt` with dependencies
    - _Requirements: 10.3, 10.5_
  
  - [x] 1.2 Implement experiment configuration
    - Create `src/config.py` with dataclass ExperimentConfig
    - Define default parameters for all models
    - Include num_simulations parameter (default 10)
    - _Requirements: 10.1_

- [x] 2. Implement Data Loader
  - [x] 2.1 Create data loading module
    - Implement `src/data_loader.py` with class DataLoader
    - Implement `load()` to load reviews dataset (Amazon or similar)
    - Implement `convert_rating_to_label()` for binary conversion
    - Implement stratified train/validation/test split (70/15/15) BEFORE preprocessing
    - Support multiple random seeds for simulations
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_
  
  - [ ]* 2.2 Write property test for label conversion
    - **Property 1: Label Conversion Correctness**
    - **Validates: Requirements 1.3, 1.4**
  
  - [ ]* 2.3 Write property test for split stratification
    - **Property 2: Train-Validation-Test Split Stratification**
    - **Validates: Requirements 1.5**

- [x] 3. Implement Preprocessor
  - [x] 3.1 Create preprocessing module
    - Implement `src/preprocessor.py` with class Preprocessor
    - Implement `fit()` to fit on training only
    - Implement `transform()` to apply transformations
    - Include: lowercase, special character removal, stopwords, tokenization
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_
  
  - [ ]* 3.2 Write property tests for preprocessing
    - **Property 3: Preprocessing Lowercase Invariant**
    - **Property 4: Preprocessing Stopword Removal**
    - **Validates: Requirements 2.2, 2.4**

- [x] 4. Checkpoint - Validate data pipeline
  - Execute notebook with loading and preprocessing
  - Verify class distribution in train/val/test
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement SVM + Bag of Words
  - [x] 5.1 Create TF-IDF vectorization module
    - Implement `src/vectorizers.py` with class BoWVectorizer
    - Wrapper for TfidfVectorizer from sklearn (max_features=5000, ngram_range=(1,2))
    - Implement `fit()` on training only, `transform()` for all sets
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 5.2 Create SVM classifier
    - Implement `src/classifiers.py` with class SVMClassifier
    - Wrapper for LinearSVC from sklearn (kernel='linear', C=1.0)
    - _Requirements: 3.4_
  
  - [ ]* 5.3 Write property tests for TF-IDF
    - **Property 5: TF-IDF Vectorization Validity**
    - **Property 6: Vocabulary Consistency for Test Data**
    - **Validates: Requirements 3.2, 3.3**

- [x] 6. Implement SVM + Embeddings
  - [x] 6.1 Create embedding encoding module
    - Implement `src/embedding_encoder.py` with class EmbeddingEncoder
    - Load GloVe via gensim
    - Implement `encode()` to calculate mean vector
    - Handle OOV with zero vector
    - _Requirements: 4.1, 4.2, 4.3_
  
  - [x] 6.2 Integrate embeddings with SVM
    - Reuse SVMClassifier with embedding vectors (kernel='rbf', gamma='scale')
    - _Requirements: 4.4_
  
  - [ ]* 6.3 Write property test for embeddings
    - **Property 7: Embedding Average Correctness**
    - **Validates: Requirements 4.2, 4.3**

- [x] 7. Checkpoint - Validate SVM models
  - Train and evaluate SVM+BoW and SVM+Embeddings
  - Compare preliminary metrics
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement BERT Classifier
  - [x] 8.1 Create BERT module
    - Implement `src/bert_classifier.py` with class BERTClassifier
    - Load DistilBERT via transformers
    - Implement `tokenize()` with truncation and padding
    - Implement `fit()` for fine-tuning with early stopping on validation
    - Implement `predict()` for inference
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [ ]* 8.2 Write property test for BERT tokenization
    - **Property 8: BERT Tokenization Length Constraint**
    - **Validates: Requirements 5.3**

- [ ] 9. Implement ICL Classifier (Bonus)
  - [ ] 9.1 Create ICL module
    - Implement `src/icl_classifier.py` with class ICLClassifier
    - Implement `build_prompt()` for few-shot (5 strategic examples)
    - Implement `classify()` and `classify_batch()`
    - Integrate with OpenAI API or alternative
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  
  - [ ]* 9.2 Write property test for ICL prompts
    - **Property 12: ICL Prompt Contains Examples**
    - **Validates: Requirements 6.2**

- [x] 10. Implement Evaluator
  - [x] 10.1 Create evaluation module
    - Implement `src/evaluator.py` with class Evaluator
    - Implement `evaluate()` to calculate F1, accuracy, precision, recall
    - Implement `get_confusion_matrix()`
    - Implement `get_comparison_table()` with mean ± std
    - Implement `get_error_examples()`
    - _Requirements: 3.5, 3.6, 4.5, 4.6, 5.7, 5.8, 6.5, 6.6, 9.1, 9.8_
  
  - [ ]* 10.2 Write property tests for evaluation
    - **Property 9: Metrics Range Validity**
    - **Property 10: Confusion Matrix Validity**
    - **Property 11: Error Examples Correctness**
    - **Validates: Requirements 3.5, 3.6, 9.8**

- [x] 11. Checkpoint - Validate all classifiers
  - Train and evaluate all 4 methods
  - Generate preliminary comparison table
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement Statistical Validator
  - [x] 12.1 Create statistical validation module
    - Implement `src/statistical_validator.py` with class StatisticalValidator
    - Implement `shapiro_normality()` for normality test
    - Implement `kruskal_wallis_multiple()` for multi-group comparison
    - Implement `wilcoxon_pairwise()` for paired comparison
    - Implement `generate_pvalue_matrix()` for all model pairs
    - Implement `generate_report()` for complete statistical report
    - Use α=0.05 significance level
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 7.7_
  
  - [ ]* 12.2 Write property tests for statistical validation
    - **Property 13: Statistical Test P-Value Range**
    - **Property 14: Simulation Results Consistency**
    - **Validates: Requirements 7.4, 7.5**

- [x] 13. Implement Multiple Simulations Runner
  - [x] 13.1 Create simulation runner
    - Implement function to run N simulations (default 10) per model
    - Vary random seed for data split and model initialization
    - Store all metrics per simulation in CSV
    - Calculate mean, std, confidence intervals
    - _Requirements: 7.1, 7.2, 7.8_

- [ ] 14. Implement Advanced NLP Analysis
  - [ ] 14.1 Create advanced analysis module
    - Implement `src/advanced_analysis.py` with class AdvancedNLPAnalysis
    - Implement `analyze_length_vs_accuracy()` with bins (0-50, 51-100, 101-200, 201-500, 500+)
    - Calculate Pearson/Spearman correlation
    - _Requirements: 8.1, 8.2_
  
  - [ ] 14.2 Implement typo robustness analysis
    - Create perturbed test dataset (5% char swaps, accent removal, letter duplication)
    - Compare accuracy on clean vs perturbed
    - Calculate degradation percentage
    - _Requirements: 8.3, 8.4_
  
  - [ ] 14.3 Implement emoji and sarcasm analysis
    - Analyze accuracy on reviews with vs without emojis
    - Identify and analyze sarcastic reviews (manual annotation of 50-100 examples)
    - _Requirements: 8.5, 8.6_
  
  - [ ] 14.4 Implement formality analysis
    - Detect formality levels (formal, informal with slang, excited with caps)
    - Analyze accuracy per formality category
    - _Requirements: 8.7_

- [ ] 15. Checkpoint - Validate statistical and advanced analysis
  - Run 10 simulations for each model
  - Generate statistical report with Wilcoxon tests
  - Generate advanced analysis results
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement Visualizer
  - [x] 16.1 Create visualization module
    - Implement `src/visualizer.py` with class Visualizer
    - Implement `plot_metrics_comparison()` - grouped bar chart
    - Implement `plot_confusion_matrix()` - heatmap
    - Implement `plot_boxplots()` - metric distribution across simulations
    - Implement `plot_line_evolution()` - metric evolution across simulations
    - Implement `plot_pvalue_matrix()` - significance matrix with color coding
    - Implement `plot_confidence_intervals()` - bars with 95% CI
    - Implement `save_all_figures()` - export PNG 300dpi
    - Use seaborn with professional theme
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.9, 9.10, 9.11_

- [ ] 17. Create main Jupyter Notebook
  - [ ] 17.1 Structure notebook with complete pipeline
    - Section 1: Imports and configuration
    - Section 2: Data loading and exploration
    - Section 3: Preprocessing
    - Section 4: SVM + BoW
    - Section 5: SVM + Embeddings
    - Section 6: BERT
    - Section 7: ICL (Bonus)
    - Section 8: Multiple simulations
    - Section 9: Statistical validation
    - Section 10: Advanced NLP analysis
    - Section 11: Comparison and visualizations
    - Section 12: Conclusions
    - _Requirements: 10.1, 10.2_
  
  - [ ] 17.2 Add documentation and comments
    - Explain each pipeline stage
    - Document design decisions
    - Include qualitative error analysis
    - _Requirements: 10.2, 9.8_

- [ ] 18. Generate final visualizations
  - [ ] 18.1 Execute complete pipeline and export results
    - Generate all figures in high resolution
    - Export comparison tables
    - Export statistical report
    - Save to `results/`
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.11_

- [ ] 19. Final checkpoint
  - Verify all exported figures
  - Review notebook for presentation
  - Verify statistical significance of results
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (property tests) and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The Jupyter notebook is the main deliverable for the course
- Figures must be ready for presentation slides
- Statistical validation with 10+ simulations ensures scientific rigor
- Advanced NLP analyses provide deeper insights into model behavior
