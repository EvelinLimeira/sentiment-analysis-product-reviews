# Interactive Prediction Demo Guide

## Overview

The Colab notebook now includes an interactive section where users can test the trained models with their own product reviews. This feature allows hands-on experimentation and demonstrates the practical application of the sentiment analysis models.

## Features

### 1. Single Review Prediction

Test any review with a specific model:

```python
predict_sentiment("Your review here", "bert")
```

**Supported models:**
- `bert` - BERT (DistilBERT) model
- `svm_bow` - SVM with Bag of Words
- `svm_embeddings` - SVM with Word Embeddings

**Output includes:**
- Predicted sentiment (POSITIVE/NEGATIVE)
- Confidence percentage
- Probability distribution for both classes

### 2. Pre-loaded Examples

Four example reviews to test model behavior:

#### Example 1: Clearly Positive
```
"This product is absolutely amazing! Best purchase I've ever made. Highly recommend!"
```

#### Example 2: Clearly Negative
```
"Terrible product. Broke after one day. Complete waste of money. Do not buy!"
```

#### Example 3: Mixed/Ambiguous
```
"The product works okay, but the price is too high for what you get."
```

#### Example 4: Sarcastic (Challenging)
```
"Oh great, another broken product. Just what I needed. Thanks a lot!"
```

### 3. Custom Input

Users can input their own reviews directly in the notebook:

```python
my_review = input("Enter your product review: ")
model_choice = input("Choose model (bert/svm_bow/svm_embeddings): ")
predict_sentiment(my_review, model_choice)
```

### 4. Batch Prediction

Analyze multiple reviews at once:

```python
reviews_to_test = [
    "Excellent quality and fast shipping!",
    "Not worth the money, very disappointed.",
    "It's okay, nothing special.",
    "Love it! Will buy again.",
    "Worst purchase ever. Returning it."
]

# Batch prediction with BERT
for review in reviews_to_test:
    prediction = classifier_bert.predict([review])[0]
    # ... display results
```

### 5. Model Comparison Widget

Compare all three models side-by-side:

```python
compare_all_models("Your review here")
```

**Output shows:**
- Prediction from each model
- Confidence scores
- Agreement/disagreement indicator

## Usage Instructions

### Step 1: Train Models First

Before using the interactive demo, you must train the models by running:
- Section 5: SVM + Bag of Words
- Section 6: SVM + Embeddings  
- Section 7: BERT Classifier

### Step 2: Navigate to Section 11

Scroll to "Section 11: Interactive Prediction Demo"

### Step 3: Run the Function Definition Cell

This creates the `predict_sentiment()` and `compare_all_models()` functions.

### Step 4: Try the Examples

Run the example cells to see how models handle different types of reviews.

### Step 5: Test Your Own Reviews

Use the custom input cells to test your own product reviews.

## Example Output

```
============================================================
Analyzing with BERT
============================================================

Review: "This product is absolutely amazing! Best purchase I've ever made."

Prediction: POSITIVE 😊
Confidence: 99.87%

Probabilities:
  Negative: 0.13%
  Positive: 99.87%

============================================================
```

## Comparing Models

When you use `compare_all_models()`, you'll see:

```
======================================================================
COMPARING ALL MODELS
======================================================================

Review: "This product exceeded my expectations! Highly recommended."

----------------------------------------------------------------------
BERT                 → POSITIVE (99.2% confidence)
SVM + BoW            → POSITIVE (92.5% confidence)
SVM + Embeddings     → POSITIVE (87.3% confidence)
----------------------------------------------------------------------

✓ All models agree: POSITIVE
======================================================================
```

## Educational Value

### For Students

- **Hands-on learning**: Test models with real examples
- **Understanding confidence**: See how certain models are about predictions
- **Model comparison**: Observe differences between approaches
- **Edge cases**: Test with sarcasm, mixed reviews, etc.

### For Instructors

- **Live demonstrations**: Show model behavior in real-time
- **Discussion prompts**: Use disagreements to discuss model limitations
- **Engagement**: Interactive elements keep students interested
- **Practical application**: Bridge theory and practice

## Common Use Cases

### 1. Testing Model Robustness

Try reviews with:
- Typos: "Grate prodct, verry gud!"
- Slang: "This thing is lit! Totally fire!"
- Formal language: "The product meets all specified requirements."
- Emojis: "Love it! ❤️😍"

### 2. Finding Model Weaknesses

Test with:
- Sarcasm: "Oh wonderful, it broke immediately. Perfect!"
- Mixed sentiment: "Good quality but terrible customer service."
- Subtle negativity: "It's fine, I guess."

### 3. Comparing Model Behavior

See how different models handle:
- Short reviews: "Great!"
- Long reviews: Multiple sentences with details
- Technical language: Product specifications
- Emotional language: Strong positive/negative words

## Tips for Best Results

### 1. Write Clear Reviews

Models work best with:
- Complete sentences
- Clear sentiment indicators
- Product-focused content

### 2. Test Edge Cases

Try challenging examples:
- Double negatives: "Not bad at all"
- Conditional statements: "Would be good if it worked"
- Comparisons: "Better than the previous version"

### 3. Observe Confidence Scores

- High confidence (>95%): Model is very certain
- Medium confidence (70-95%): Model is fairly certain
- Low confidence (<70%): Model is uncertain, review may be ambiguous

### 4. Compare Models

- BERT typically has higher confidence
- SVM models may disagree on ambiguous reviews
- Agreement across all models indicates clear sentiment

## Troubleshooting

### Error: "Model must be fitted before prediction"

**Solution:** Run the training cells (Sections 5-7) first.

### Error: "NameError: name 'classifier_bert' is not defined"

**Solution:** Train the BERT model in Section 7.

### Error: "NameError: name 'preprocessor' is not defined"

**Solution:** Run Section 4 (Text Preprocessing) first.

### Unexpected Predictions

**Possible reasons:**
- Sarcasm or irony (challenging for all models)
- Mixed sentiment (positive and negative aspects)
- Domain-specific language
- Very short or very long reviews

## Advanced Usage

### Custom Preprocessing

Test how preprocessing affects predictions:

```python
# Original text
original = "This product is AMAZING!!!"

# Without preprocessing (BERT)
predict_sentiment(original, 'bert')

# With preprocessing (SVM)
predict_sentiment(original, 'svm_bow')
```

### Confidence Threshold Analysis

Analyze predictions by confidence level:

```python
reviews = [...]  # Your reviews
low_confidence = []

for review in reviews:
    probs = classifier_bert.predict_proba([review])[0]
    confidence = max(probs)
    
    if confidence < 0.7:
        low_confidence.append((review, confidence))

print(f"Found {len(low_confidence)} low-confidence predictions")
```

### Error Analysis

Find where models disagree:

```python
def find_disagreements(reviews):
    disagreements = []
    
    for review in reviews:
        bert_pred = classifier_bert.predict([review])[0]
        svm_pred = classifier_bow.predict(vectorizer.transform([review]))[0]
        
        if bert_pred != svm_pred:
            disagreements.append(review)
    
    return disagreements
```

## Integration with Course Material

### Lecture Topics

- **Preprocessing impact**: Compare BERT (no preprocessing) vs SVM (with preprocessing)
- **Model complexity**: Discuss why BERT is more confident
- **Feature engineering**: Show how BoW vs Embeddings differ
- **Evaluation metrics**: Connect predictions to accuracy/F1-score

### Assignments

Suggested exercises:
1. Find 5 reviews where models disagree
2. Test 10 sarcastic reviews and analyze results
3. Compare confidence scores across models
4. Create a confusion matrix from custom reviews

### Research Questions

- How does review length affect confidence?
- Which model handles negation better?
- Do models agree more on extreme sentiments?
- How does emoji usage impact predictions?

## Best Practices

### For Demonstrations

1. **Start simple**: Use clearly positive/negative examples
2. **Build complexity**: Progress to ambiguous cases
3. **Show failures**: Demonstrate model limitations
4. **Encourage exploration**: Let students test their own examples

### For Evaluation

1. **Document predictions**: Save interesting examples
2. **Analyze patterns**: Look for systematic errors
3. **Compare with ground truth**: Use known sentiments
4. **Discuss results**: Explain why models succeed/fail

## Future Enhancements

Potential additions:
- Visualization of attention weights (BERT)
- Feature importance display (SVM)
- Confidence calibration plots
- Real-time sentiment distribution
- Export predictions to CSV

## References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [SVM for Text Classification](https://scikit-learn.org/stable/modules/svm.html)
- [Sentiment Analysis Best Practices](https://www.nltk.org/howto/sentiment.html)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main [README](../../README.md)
3. See [Colab Troubleshooting Guide](colab-troubleshooting.md)
4. Open an issue on GitHub

---

**Last Updated:** February 2026
