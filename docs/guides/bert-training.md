# BERT Training Configuration

## Overview

This document describes the optimized BERT training configuration used in this project for sentiment analysis on product reviews.

## Configuration

### Model
- **Base Model**: `distilbert-base-uncased`
- **Architecture**: DistilBERT (distilled version of BERT)
- **Parameters**: ~66M parameters
- **Max Sequence Length**: 512 tokens

### Training Parameters

```python
{
    'num_epochs': 10,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'patience': 3,
    'early_stopping': True
}
```

### Key Features

#### 1. Extended Training (10 Epochs)
- Increased from 3 to 10 epochs for better convergence
- Allows model to learn more complex patterns
- Typical convergence occurs between epochs 4-7

#### 2. Early Stopping
- **Patience**: 3 epochs
- Monitors validation F1-score
- Stops training if no improvement for 3 consecutive epochs
- Prevents overfitting and saves computation time
- Automatically restores best model weights

#### 3. Optimized Batch Size
- **Batch size**: 32
- Balanced for GPU memory utilization
- Provides stable gradient estimates
- Suitable for 8GB+ VRAM GPUs

#### 4. Learning Rate
- **Rate**: 2e-5 (0.00002)
- Standard for BERT fine-tuning
- Uses AdamW optimizer
- No learning rate scheduling (constant rate)

## Training Process

### Typical Training Run

```
Epoch 1/10: val_f1=0.8523 (baseline)
Epoch 2/10: val_f1=0.8891 (improvement)
Epoch 3/10: val_f1=0.9124 (improvement)
Epoch 4/10: val_f1=0.9156 (improvement) ← Best
Epoch 5/10: val_f1=0.9142 (no improvement, patience=1)
Epoch 6/10: val_f1=0.9138 (no improvement, patience=2)
Epoch 7/10: val_f1=0.9135 (no improvement, patience=3)
→ Early stopping triggered
→ Restored weights from Epoch 4
```

### Training Time
- **Per Epoch**: ~50-60 seconds (with GPU)
- **Total Training**: ~5-8 minutes (with early stopping)
- **Per Simulation**: ~8.6 minutes average (including data loading)

## Results

### Performance Metrics (10 Simulations)
- **Accuracy**: 91.58% ± 0.80%
- **F1-Score**: 91.58% ± 0.80%
- **Precision**: 91.61% ± 0.79%
- **Recall**: 91.58% ± 0.79%

### Comparison with Previous Configuration

| Configuration | Epochs | Early Stop | Batch | Accuracy | F1-Score |
|---------------|--------|------------|-------|----------|----------|
| **Current** | 10 | Yes (p=3) | 32 | 91.58% | 91.58% |
| Previous | 3 | No | 16 | ~88-89% | ~88-89% |

**Improvement**: +2-3% in both accuracy and F1-score

## Hardware Requirements

### Minimum
- **GPU**: 6GB VRAM 
- **RAM**: 8GB system RAM
- **Storage**: 2GB for model cache

### Recommended
- **GPU**: 8GB+ VRAM 
- **RAM**: 16GB system RAM
- **Storage**: 5GB for model cache and results

### Without GPU
- Training is possible but **very slow** (~30-40x slower)
- Not recommended for 10 simulations
- Consider using Google Colab with free GPU

## Usage

### Basic Training

```python
from src.bert_classifier import BERTClassifier

# Initialize with optimized configuration
classifier = BERTClassifier(
    model_name='distilbert-base-uncased',
    batch_size=32,
    num_epochs=10,
    patience=3
)

# Train with early stopping
classifier.fit(
    train_texts, train_labels,
    val_texts, val_labels
)

# Best model is automatically loaded
predictions = classifier.predict(test_texts)
```

### Custom Configuration

```python
# Adjust for different hardware
classifier = BERTClassifier(
    model_name='distilbert-base-uncased',
    batch_size=16,  # Reduce for less VRAM
    num_epochs=15,  # Increase max epochs
    patience=5      # More patience
)
```

## Best Practices

1. **Always use validation set** for early stopping
2. **Monitor GPU memory** - reduce batch size if OOM errors occur
3. **Use mixed precision** (FP16) for faster training on modern GPUs
4. **Save best model** after training completes
5. **Log training metrics** for analysis

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size to 16 or 8
- Reduce max sequence length to 256
- Use gradient accumulation

### Slow Training
- Ensure GPU is being used (check with `torch.cuda.is_available()`)
- Close other GPU-intensive applications
- Consider using Google Colab

### Poor Performance
- Increase number of epochs
- Adjust learning rate (try 3e-5 or 5e-5)
- Check data quality and preprocessing

## References

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
3. Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/

## See Also

- [Quick Start Guide](quick-start.md)
- [Main README](../../README.md)
- [Experiment Report](../../EXPERIMENT_REPORT.md)
