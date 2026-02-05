"""
Text perturbation module for robustness testing.

This module provides utilities to create perturbed versions of text data
for testing model robustness to typos, character swaps, and other text
corruptions.

"""

import random
import re
import unicodedata
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPerturbation:
    """
    Creates perturbed versions of text for robustness testing.
    
    This class implements various text perturbation strategies including:
    - Character swaps (adjacent character transposition)
    - Accent removal (diacritic stripping)
    - Letter duplication
    
    Attributes:
        perturbation_rate: Probability of perturbing each word (default 5%)
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, perturbation_rate: float = 0.05, random_state: int = 42):
        """
        Initialize the TextPerturbation.
        
        Args:
            perturbation_rate: Probability of perturbing each word (default 5%)
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If perturbation_rate is not between 0 and 1
        """
        if not 0 <= perturbation_rate <= 1:
            raise ValueError(
                f"perturbation_rate must be between 0 and 1, got {perturbation_rate}"
            )
        
        self.perturbation_rate = perturbation_rate
        self.random_state = random_state
        random.seed(random_state)
        
        logger.info(
            f"TextPerturbation initialized with rate={perturbation_rate}, "
            f"seed={random_state}"
        )
    
    def swap_adjacent_chars(self, word: str) -> str:
        """
        Swap two adjacent characters in a word.
        
        Args:
            word: Input word
            
        Returns:
            Word with adjacent characters swapped, or original if too short
        """
        if len(word) < 2:
            return word
        
        # Choose a random position to swap (not the last character)
        pos = random.randint(0, len(word) - 2)
        
        # Swap characters at pos and pos+1
        word_list = list(word)
        word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
        
        return ''.join(word_list)
    
    def remove_accents(self, text: str) -> str:
        """
        Remove accents/diacritics from text.
        
        Converts accented characters to their base form (e.g., é -> e).
        
        Args:
            text: Input text
            
        Returns:
            Text with accents removed
        """
        # Normalize to NFD (decomposed form)
        nfd = unicodedata.normalize('NFD', text)
        
        # Filter out combining characters (accents)
        without_accents = ''.join(
            char for char in nfd 
            if unicodedata.category(char) != 'Mn'
        )
        
        return without_accents
    
    def duplicate_letter(self, word: str) -> str:
        """
        Duplicate a random letter in a word.
        
        Args:
            word: Input word
            
        Returns:
            Word with a duplicated letter, or original if empty
        """
        if len(word) == 0:
            return word
        
        # Choose a random position
        pos = random.randint(0, len(word) - 1)
        
        # Duplicate the character at that position
        return word[:pos + 1] + word[pos] + word[pos + 1:]
    
    def perturb_word(self, word: str) -> str:
        """
        Apply a random perturbation to a word.
        
        Randomly chooses one of: character swap, accent removal, or letter duplication.
        
        Args:
            word: Input word
            
        Returns:
            Perturbed word
        """
        if len(word) < 2:
            return word
        
        # Choose perturbation type randomly
        perturbation_type = random.choice(['swap', 'accent', 'duplicate'])
        
        if perturbation_type == 'swap':
            return self.swap_adjacent_chars(word)
        elif perturbation_type == 'accent':
            return self.remove_accents(word)
        else:  # duplicate
            return self.duplicate_letter(word)
    
    def perturb_text(self, text: str) -> str:
        """
        Perturb a text by applying random perturbations to words.
        
        Each word has a probability of perturbation_rate of being perturbed.
        
        Args:
            text: Input text
            
        Returns:
            Perturbed text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Tokenize by whitespace while preserving punctuation
        words = text.split()
        
        perturbed_words = []
        for word in words:
            # Check if we should perturb this word
            if random.random() < self.perturbation_rate:
                # Separate word from trailing punctuation
                match = re.match(r'^(\W*)(\w+)(\W*)$', word)
                if match:
                    prefix, core_word, suffix = match.groups()
                    perturbed_core = self.perturb_word(core_word)
                    perturbed_words.append(prefix + perturbed_core + suffix)
                else:
                    # If pattern doesn't match, just perturb the whole thing
                    perturbed_words.append(self.perturb_word(word))
            else:
                perturbed_words.append(word)
        
        return ' '.join(perturbed_words)
    
    def perturb_texts(self, texts: List[str]) -> List[str]:
        """
        Perturb a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of perturbed texts
        """
        logger.info(f"Perturbing {len(texts)} texts with rate={self.perturbation_rate}")
        
        perturbed = [self.perturb_text(text) for text in texts]
        
        # Calculate statistics
        total_words = sum(len(text.split()) for text in texts)
        changed_texts = sum(1 for orig, pert in zip(texts, perturbed) if orig != pert)
        
        logger.info(
            f"Perturbed {changed_texts}/{len(texts)} texts "
            f"({changed_texts/len(texts)*100:.1f}%)"
        )
        
        return perturbed
    
    def get_perturbation_examples(
        self, 
        texts: List[str], 
        n_examples: int = 5
    ) -> List[Tuple[str, str]]:
        """
        Get examples of original and perturbed texts.
        
        Args:
            texts: List of input texts
            n_examples: Number of examples to return
            
        Returns:
            List of (original, perturbed) tuples
        """
        # Perturb texts
        perturbed = self.perturb_texts(texts)
        
        # Find texts that were actually changed
        changed_pairs = [
            (orig, pert) 
            for orig, pert in zip(texts, perturbed) 
            if orig != pert
        ]
        
        # Sample n_examples
        n_samples = min(n_examples, len(changed_pairs))
        if n_samples == 0:
            return []
        
        sampled_indices = random.sample(range(len(changed_pairs)), n_samples)
        examples = [changed_pairs[i] for i in sampled_indices]
        
        return examples


def create_perturbed_dataset(
    texts: List[str],
    labels: List[int],
    perturbation_rate: float = 0.05,
    random_state: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Create a perturbed version of a dataset.
    
    Convenience function to perturb texts while keeping labels unchanged.
    
    Args:
        texts: List of input texts
        labels: List of labels (unchanged)
        perturbation_rate: Probability of perturbing each word (default 5%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (perturbed_texts, labels)
    """
    perturber = TextPerturbation(
        perturbation_rate=perturbation_rate,
        random_state=random_state
    )
    
    perturbed_texts = perturber.perturb_texts(texts)
    
    return perturbed_texts, labels


# Example usage
if __name__ == "__main__":
    # Example texts
    example_texts = [
        "This product is absolutely amazing! I love it.",
        "Terrible quality, would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
        "Best purchase I've made this year, highly recommended!",
        "Disappointed with the quality, expected much better."
    ]
    
    # Create perturbation object
    perturber = TextPerturbation(perturbation_rate=0.1, random_state=42)
    
    # Get examples
    print("="*80)
    print("TEXT PERTURBATION EXAMPLES")
    print("="*80)
    
    examples = perturber.get_perturbation_examples(example_texts, n_examples=5)
    
    for i, (original, perturbed) in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Original:  {original}")
        print(f"  Perturbed: {perturbed}")
    
    print("\n" + "="*80)
