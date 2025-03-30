from abc import ABC, abstractmethod
import random
from nltk.corpus import wordnet

class INoiseStrategy(ABC):
    @abstractmethod
    def apply(self, documents, noise_level):
        """Apply noise to the pandas series."""
        pass



class AddRandomWordsNoise(INoiseStrategy):
    def apply(self, documents, noise_level=0.1):
        probability = noise_level
        # Preload synsets to avoid repeated expensive calls
        synsets = list(wordnet.all_synsets())

        def get_random_wordnet_word():
            """Get a random word from WordNet, ensuring it's a single-word noun or verb."""
            while True:
                random_synset = random.choice(synsets)  # Avoid regenerating the list
                word = random_synset.lemmas()[0].name()  # Get first lemma of the synset
                if '_' not in word:  # Ensure single-word output
                    return word
                
        words = documents.split()
        modified_words = []
        
        for word in words:
            modified_words.append(word)
            if random.random() < probability:
                random_word = get_random_wordnet_word()
                modified_words.append(random_word)
        
        return ' '.join(modified_words)


class AddRandomCharsNoise(INoiseStrategy):
    def apply(self, documents, noise_level=0.1):
        probability = noise_level
        words = documents.split()
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


class DeleteRandomWordsNoise(INoiseStrategy):
    def apply(self, documents, noise_level=0.1):
        probability = noise_level
        words = documents.split()
        if len(words) <= 3:  # Don't delete if text is too short
            return documents
        
        modified_words = []
        
        for word in words:
            if random.random() >= probability:
                modified_words.append(word)
        
        # Ensure we don't delete all words
        if not modified_words:
            modified_words = [random.choice(words)]
        
        return ' '.join(modified_words)