# Requirements Document

## Introduction

Sentiment analysis system for product reviews as part of a Natural Language Processing (NLP) course. The project implements and compares different sentiment classification approaches: SVM with Bag of Words, SVM with Embeddings, and BERT, plus a bonus with in-context learning. The goal is to evaluate each method's performance using F1-score and accuracy metrics, with rigorous statistical validation through multiple simulation runs. The project prioritizes simple and fast solutions as per the professor's guidance while maintaining scientific rigor.

## Glossary

- **Pipeline**: The complete processing flow from data collection to evaluation
- **Data_Loader**: Component responsible for loading and preparing the dataset
- **Preprocessor**: Component that cleans and normalizes texts
- **Classifier**: Machine learning model that categorizes sentiments
- **Evaluator**: Component that calculates performance metrics
- **Review**: Product review text with associated rating
- **Sentiment_Label**: Binary classification (positive/negative) derived from rating
- **BoW_Vectorizer**: Component that converts text to Bag of Words representation
- **Embedding_Encoder**: Component that converts text to embedding vectors
- **BERT_Classifier**: Classifier based on pre-trained BERT model
- **ICL_Classifier**: Classifier using In-Context Learning with LLM
- **F1_Score**: Harmonic mean between precision and recall
- **Accuracy**: Proportion of correct predictions over total
- **Statistical_Validator**: Component that performs statistical significance tests
- **Simulation**: A single training/evaluation run with a specific random seed

## Requirements

### Requirement 1: Data Collection and Preparation

**User Story:** As a researcher, I want to load and prepare product reviews with text and rating, to have balanced training, validation, and test data without data leakage.

#### Acceptance Criteria

1. THE Data_Loader SHALL load a public product reviews dataset with balanced positive and negative reviews (minimum 3,000 reviews, ideal 5,000+)
2. WHEN the dataset is loaded, THE Data_Loader SHALL extract the review text and associated numeric rating
3. THE Data_Loader SHALL convert numeric ratings to binary Sentiment_Label where ratings 1-2 are negative and ratings 4-5 are positive
4. WHEN data is converted, THE Data_Loader SHALL discard reviews with rating 3 to avoid ambiguity
5. THE Data_Loader SHALL split data into train (70%), validation (15%), and test (15%) sets with stratification by class BEFORE any preprocessing
6. THE Data_Loader SHALL ensure no information from validation or test sets is used during training or hyperparameter tuning
7. THE Data_Loader SHALL report class distribution in train, validation, and test sets
8. THE Data_Loader SHALL support multiple random seeds for reproducible experiments across simulations

### Requirement 2: Text Preprocessing

**User Story:** As a researcher, I want to preprocess texts in a standardized way without data leakage, to ensure consistency and validity of results.

#### Acceptance Criteria

1. THE Preprocessor SHALL be fitted ONLY on the training set to avoid data leakage
2. THE Preprocessor SHALL convert all text to lowercase
3. THE Preprocessor SHALL remove special characters and excessive punctuation
4. THE Preprocessor SHALL remove stopwords from the dataset language using a predefined list
5. THE Preprocessor SHALL apply word tokenization for SVM models
6. WHEN processing validation or test sets, THE Preprocessor SHALL use ONLY transformations learned from training
7. THE Preprocessor SHALL handle unknown words (OOV) consistently
8. THE Preprocessor SHALL NOT apply traditional preprocessing for BERT, which uses its own tokenizer

### Requirement 3: SVM with Bag of Words Classifier

**User Story:** As a researcher, I want to train an SVM classifier using Bag of Words, to have a traditional and fast baseline.

#### Acceptance Criteria

1. THE BoW_Vectorizer SHALL be fitted ONLY on the training set
2. THE BoW_Vectorizer SHALL convert preprocessed texts to TF-IDF vectors (max_features=5000, ngram_range=1-2)
3. WHEN processing validation or test sets, THE BoW_Vectorizer SHALL use ONLY the vocabulary learned from training
4. THE Classifier SHALL train an SVM model with linear kernel (C=1.0) using TF-IDF vectors from training
5. WHEN the model is trained, THE Evaluator SHALL calculate F1_Score and Accuracy on the test set
6. THE Evaluator SHALL generate confusion matrix for the SVM-BoW classifier
7. THE Pipeline SHALL save the trained model and vectorizer for reproducibility

### Requirement 4: SVM with Embeddings Classifier

**User Story:** As a researcher, I want to train an SVM classifier using pre-trained embeddings, to compare dense representations with BoW.

#### Acceptance Criteria

1. THE Embedding_Encoder SHALL load pre-trained Word2Vec or GloVe embeddings (trained on external corpus)
2. WHEN text is processed, THE Embedding_Encoder SHALL calculate the mean vector of embeddings for all words in the document (optionally TF-IDF weighted)
3. THE Embedding_Encoder SHALL handle out-of-vocabulary (OOV) words by ignoring them or using zero vector
4. THE Classifier SHALL train an SVM model with RBF kernel (C=1.0, gamma='scale') using embedding vectors from training
5. WHEN the model is trained, THE Evaluator SHALL calculate F1_Score and Accuracy on the test set
6. THE Evaluator SHALL generate confusion matrix for the SVM-Embedding classifier

### Requirement 5: BERT Classifier

**User Story:** As a researcher, I want to use BERT for sentiment classification, to compare with traditional methods.

#### Acceptance Criteria

1. THE BERT_Classifier SHALL load a DistilBERT-base model (66M parameters, ~250MB) suitable for 8GB GPU
2. THE BERT_Classifier SHALL use native WordPiece/BPE tokenization without traditional preprocessing
3. THE BERT_Classifier SHALL tokenize texts using the corresponding tokenizer with truncation (max 512 tokens) and padding
4. THE BERT_Classifier SHALL fine-tune the model ONLY on the training set with appropriate batch size for 8GB VRAM
5. THE BERT_Classifier SHALL use validation set for early stopping
6. IF computational resources are limited, THEN THE BERT_Classifier SHALL use gradient accumulation or smaller batch size
7. WHEN the model is trained, THE Evaluator SHALL calculate F1_Score and Accuracy on the test set
8. THE Evaluator SHALL generate confusion matrix for the BERT_Classifier

### Requirement 6: Bonus - In-Context Learning

**User Story:** As a researcher, I want to use in-context learning with LLM, to explore zero-shot and few-shot classification.

#### Acceptance Criteria

1. THE ICL_Classifier SHALL connect to an available LLM API (OpenAI GPT, Claude, or Gemini)
2. THE ICL_Classifier SHALL build prompts with 5 strategic few-shot examples of positive and negative reviews
3. WHEN classifying, THE ICL_Classifier SHALL send the prompt with the review to be classified
4. THE ICL_Classifier SHALL classify a representative sample of the test set
5. WHEN classification is complete, THE Evaluator SHALL calculate F1_Score and Accuracy on the sample
6. THE Evaluator SHALL document API cost and inference time

### Requirement 7: Statistical Validation with Multiple Simulations

**User Story:** As a researcher, I want to run multiple simulations with different seeds and perform statistical tests, to ensure results are statistically significant.

#### Acceptance Criteria

1. THE Pipeline SHALL execute minimum 10 simulations (ideal 30) per model with different random seeds
2. FOR each simulation, THE Evaluator SHALL extract: Accuracy, Precision (macro and per-class), Recall (macro and per-class), F1-Score (macro and weighted), training time, inference time
3. THE Statistical_Validator SHALL perform Shapiro-Wilk normality test on metric distributions
4. THE Statistical_Validator SHALL perform Kruskal-Wallis H-test to determine if there are significant differences among all models
5. THE Statistical_Validator SHALL perform Wilcoxon Signed-Rank test (paired) for pairwise model comparisons
6. THE Statistical_Validator SHALL use significance level α=0.05 (95% confidence)
7. THE Statistical_Validator SHALL generate p-value matrix for all model pairs
8. THE Pipeline SHALL store all simulation results in CSV format for analysis

### Requirement 8: Advanced NLP Analysis

**User Story:** As a researcher, I want to perform advanced NLP-specific analyses, to understand model behavior in different scenarios.

#### Acceptance Criteria

1. THE Evaluator SHALL analyze accuracy by text length bins (0-50, 51-100, 101-200, 201-500, 500+ characters)
2. THE Evaluator SHALL calculate Pearson/Spearman correlation between text length and accuracy per model
3. THE Evaluator SHALL analyze robustness to typos by creating a perturbed test dataset (5% character swaps, accent removal, letter duplication)
4. THE Evaluator SHALL compare accuracy on clean vs perturbed datasets and calculate degradation percentage
5. THE Evaluator SHALL analyze impact of emojis by comparing accuracy on reviews with vs without emojis
6. THE Evaluator SHALL identify and analyze sarcastic/ironic reviews (50-100 manually annotated examples)
7. THE Evaluator SHALL analyze sensitivity to formality/dialect (formal, informal with slang, excited with caps)

### Requirement 9: Comparison and Results Analysis

**User Story:** As a researcher, I want to compare all methods with professional visualizations, to present well-founded conclusions in data science style.

#### Acceptance Criteria

1. THE Evaluator SHALL generate styled comparison table with F1_Score and Accuracy of all classifiers (mean ± std)
2. THE Evaluator SHALL generate grouped bar chart comparing F1 and Accuracy with professional color palette
3. THE Evaluator SHALL generate styled heatmap confusion matrices for each classifier
4. THE Evaluator SHALL generate boxplots showing metric distribution across simulations for each model
5. THE Evaluator SHALL generate line plots showing metric evolution across simulations
6. THE Evaluator SHALL generate p-value significance matrix with color coding (green p<0.05, red p≥0.05)
7. THE Evaluator SHALL generate bar charts with 95% confidence intervals
8. THE Evaluator SHALL identify and document error examples from each classifier with qualitative analysis
9. THE Evaluator SHALL record training and inference time for each method in comparative chart
10. THE Evaluator SHALL use modern visual style (seaborn/plotly) with consistent theme
11. WHEN analysis is complete, THE Pipeline SHALL export all visualizations in high-resolution PNG format for slides

### Requirement 10: Documentation and Reproducibility

**User Story:** As a student, I want to document the entire process in Jupyter notebook, to submit the assignment and allow reproduction.

#### Acceptance Criteria

1. THE Pipeline SHALL be implemented in Jupyter notebook with documented cells
2. THE Pipeline SHALL include explanatory comments at each process stage
3. THE Pipeline SHALL list all libraries and versions at the beginning of the notebook
4. THE Pipeline SHALL export figures in adequate resolution for presentation
5. THE Pipeline SHALL include execution instructions in the project README
