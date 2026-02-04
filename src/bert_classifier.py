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
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        patience: int = 2,
        device: str = None
    ):
        """
        Initialize BERT classifier.
        
        Args:
            model_name: HuggingFace model name (default: distilbert-base-uncased)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for training and inference
            learning_rate: Learning rate for fine-tuning
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience (epochs without improvement)
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
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}'):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                # Save best model state
                self.best_model_state = self.model.state_dict()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
        # Load best model state
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
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
