"""
BERT Classifier for sentiment analysis.

This module implements a DistilBERT-based classifier for binary sentiment classification.
It handles tokenization, fine-tuning with early stopping, and inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import logging
import warnings
import os

# Suppress transformers warnings and info messages
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""
    
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]):
        """
        Args:
            encodings: Dictionary with 'input_ids' and 'attention_mask'
            labels: List of binary labels (0 or 1)
        """
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BERTClassifier:
    """Sentiment classifier using DistilBERT."""
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        max_length: int = 512,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        patience: int = 3,
        device: str = None
    ):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: HuggingFace model name (default: distilbert-base-uncased)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for training and inference (default: 32)
            learning_rate: Learning rate for fine-tuning
            num_epochs: Maximum number of training epochs (default: 10)
            patience: Early stopping patience (epochs without improvement, default: 3)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Model will be initialized during fit
        self.model = None
        self.is_fitted = False
    
    def tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes texts using BERT tokenizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings
    
    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int]
    ) -> 'BERTClassifier':
        """
        Fine-tunes BERT with early stopping on validation set.
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0 or 1)
            val_texts: List of validation texts
            val_labels: List of validation labels (0 or 1)
            
        Returns:
            self for method chaining
        """
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        )
        self.model.to(self.device)
        
        # Tokenize data
        train_encodings = self.tokenize(train_texts)
        val_encodings = self.tokenize(val_texts)
        
        # Create datasets
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_epoch = 0
        
        print(f"\nStarting training for {self.num_epochs} epochs (early stopping patience: {self.patience})")
        print(f"Device: {self.device} | Batch size: {self.batch_size} | Learning rate: {self.learning_rate}")
        print("-" * 80)
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Progress bar for training
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}', 
                       bar_format='{desc} {n_fmt}/{total_fmt} {bar} {elapsed} - loss: {postfix[0]:.4f} - accuracy: {postfix[1]:.4f}',
                       postfix=[0.0, 0.0])
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=1)
                train_correct += (predictions == batch['labels']).sum().item()
                train_total += batch['labels'].size(0)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                # Update progress bar
                avg_loss = train_loss / (pbar.n + 1)
                avg_acc = train_correct / train_total
                pbar.postfix = [avg_loss, avg_acc]
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    val_correct += (predictions == batch['labels']).sum().item()
                    val_total += batch['labels'].size(0)
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # Print epoch summary
            improvement_marker = ""
            if avg_val_loss < best_val_loss:
                improvement_marker = " ✓ [BEST]"
            
            print(f'Epoch {epoch + 1}/{self.num_epochs} - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}{improvement_marker}')
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_epoch = epoch + 1
                # Save best model state
                self.best_model_state = self.model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f'\n⚠ Early stopping triggered after {epoch + 1} epochs')
                    print(f'  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')
                    print(f'  No improvement for {self.patience} consecutive epochs')
                    break
        
        # Load best model state
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f'\n✓ Loaded best model from epoch {best_epoch}')
        
        print("-" * 80)
        
        self.is_fitted = True
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predicts labels for new texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenize(texts)
        
        # Create dataset and dataloader
        # Use dummy labels for prediction
        dummy_labels = [0] * len(texts)
        dataset = SentimentDataset(encodings, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device (exclude labels)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predicts class probabilities for new texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of shape (n_samples, 2) with class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        self.model.eval()
        
        # Tokenize texts
        encodings = self.tokenize(texts)
        
        # Create dataset and dataloader
        dummy_labels = [0] * len(texts)
        dataset = SentimentDataset(encodings, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities.extend(probs)
        
        return np.array(probabilities)
    
    def save_model(self, save_dir: str) -> None:
        """
        Save the trained model and tokenizer.
        
        Args:
            save_dir: Directory to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving. Call fit() first.")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model and tokenizer using HuggingFace's save_pretrained
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save additional config
        import json
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'patience': self.patience
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load_model(cls, load_dir: str, device: str = None):
        """
        Load a saved model.
        
        Args:
            load_dir: Directory containing the saved model
            device: Device to load model on (None for auto-detect)
            
        Returns:
            BERTClassifier instance with loaded model
        """
        import json
        import os
        
        # Load config
        with open(os.path.join(load_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create instance
        classifier = cls(
            model_name=config['model_name'],
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            patience=config['patience'],
            device=device
        )
        
        # Load model and tokenizer
        classifier.model = DistilBertForSequenceClassification.from_pretrained(load_dir)
        classifier.model.to(classifier.device)
        classifier.tokenizer = DistilBertTokenizer.from_pretrained(load_dir)
        classifier.is_fitted = True
        
        return classifier

