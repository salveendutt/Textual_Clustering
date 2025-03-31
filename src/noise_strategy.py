from abc import ABC, abstractmethod
import random
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from tools import get_synonyms, get_antonyms

class INoiseStrategy(ABC):
    @abstractmethod
    def _apply_per_row(self, documents, noise_level):
        """Apply noise to the pandas series."""
        pass
    
    def apply(self, documents):
        return documents.apply(self._apply_per_row)


class NoNoise(INoiseStrategy):
    def _apply_per_row(self, text):
        return text
    
    def apply(self, documents):
        return documents
    


class AddRandomCharsNoise(INoiseStrategy):
    def _apply_per_row(self, text, noise_level=0.1):
        probability = noise_level
        words = text.split()
        modified_words = []
        
        for word in words:
            if len(word) > 2 and random.random() < probability:
                # Choose a random position to insert the character
                position = random.randint(1, len(word) - 1)
                # Choose a random lowercase letter
                random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:position] + random_char + word[position:]
            modified_words.append(word)
        
        return ' '.join(modified_words)
    
class AddRandomWordsNoise(INoiseStrategy):
    def __init__(self):
        # Preload synsets once during initialization
        self.single_word_list = []
        synsets = list(wordnet.all_synsets())
        
        # Pre-filter eligible words (do this work once)
        for synset in synsets:
            word = synset.lemmas()[0].name()
            if '_' not in word:
                self.single_word_list.append(word)
                
        # Raise error if no valid words found
        if not self.single_word_list:
            raise ValueError("No valid single words found in WordNet")
    
    def _apply_per_row(self, text, noise_level=0.1):
        words = text.split()
        result = list(words)  # Copy words list
        
        # Calculate how many words to add based on probability
        num_insertions = int(len(words) * noise_level)
        
        # Add random words at random positions
        for _ in range(num_insertions):
            # Get random position to insert
            insert_pos = random.randint(0, len(result))
            # Get random word from pre-filtered list
            random_word = random.choice(self.single_word_list)
            # Insert at position
            result.insert(insert_pos, random_word)
            
        return ' '.join(result)
    

class DeleteRandomWordsNoise(INoiseStrategy):
    def _apply_per_row(self, text, noise_level=0.1):
        probability = noise_level
        words = text.split()
        if len(words) <= 3:  # Don't delete if text is too short
            return text
        
        modified_words = []
        
        for word in words:
            if random.random() >= probability:
                modified_words.append(word)
        
        # Ensure we don't delete all words
        if not modified_words:
            modified_words = [random.choice(words)]
        
        return ' '.join(modified_words)

class ShuffleSentencesNoise(INoiseStrategy):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def _apply_per_row(self, text):
        probability = self.noise_level

        if not isinstance(text, str):
            return text
        
        if random.random() < probability:
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                random.shuffle(sentences)
            return ' '.join(sentences)
        
        return text
    
class ReplaceWithSynonymsNoise(INoiseStrategy):
    def __init__(self):
        # Cache for synonyms to avoid repeated lookups
        self.synonym_cache = {}
        
    def _apply_per_row(self, text, noise_level=0.1):
        if not isinstance(text, str):
            return text
            
        words = word_tokenize(text)
        
        # Find eligible words (alphabetic and longer than 3 chars)
        eligible_indices = [i for i, word in enumerate(words) 
                           if word.isalpha() and len(word) > 3]
        
        if not eligible_indices:
            return text
            
        # Determine number of words to replace
        num_to_replace = max(1, int(len(eligible_indices) * noise_level))
        indices_to_replace = random.sample(eligible_indices, 
                                          min(num_to_replace, len(eligible_indices)))
        
        # Only modify selected words
        for idx in indices_to_replace:
            word = words[idx]
            
            # Check cache first
            if word in self.synonym_cache:
                synonyms = self.synonym_cache[word]
            else:
                synonyms = get_synonyms(word)
                self.synonym_cache[word] = synonyms
                
            if synonyms:
                words[idx] = random.choice(synonyms)
                
        return ' '.join(words)


class ReplaceWithAntonymsNoise(INoiseStrategy):
    def __init__(self):
        # Cache for antonyms to avoid repeated lookups
        self.antonym_cache = {}
        
    def _apply_per_row(self, text, noise_level=0.1):
        if not isinstance(text, str):
            return text
            
        words = word_tokenize(text)
        
        # Find eligible words (alphabetic and longer than 2 chars)
        eligible_indices = [i for i, word in enumerate(words) 
                           if word.isalpha() and len(word) > 2]
        
        if not eligible_indices:
            return text
            
        # Determine number of words to replace
        num_to_replace = max(1, int(len(eligible_indices) * noise_level))
        indices_to_replace = random.sample(eligible_indices, 
                                          min(num_to_replace, len(eligible_indices)))
        
        # Only modify selected words
        for idx in indices_to_replace:
            word = words[idx]
            word_lower = word.lower()
            
            # Check cache first
            if word_lower in self.antonym_cache:
                antonyms = self.antonym_cache[word_lower]
            else:
                antonyms = get_antonyms(word_lower)
                self.antonym_cache[word_lower] = antonyms
                
            if antonyms:
                # Use WordNet antonym
                replacement = random.choice(antonyms)
                # Keep the original capitalization pattern
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
                
        return ' '.join(words)

class CompositeNoise(INoiseStrategy):
    """Applies multiple noise strategies sequentially."""
    def __init__(self, noise_strategies):
        self.noise_strategies = noise_strategies

    def _apply_per_row(self, text):
        pass
    
    def apply(self, documents):
        for strategy in self.noise_strategies:
            documents = strategy.apply(documents)
        return documents
