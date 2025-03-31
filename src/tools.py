import re
import nltk
import string
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.casual import casual_tokenize
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS)

PUNC = string.punctuation

def process_text(text):
    text = casual_tokenize(text)
    text = [each.lower() for each in text]
    text = [re.sub('[0-9]+', '', each) for each in text]
    text = [SnowballStemmer('english').stem(each) for each in text]
    text = [w for w in text if w not in PUNC]
    text = [w for w in text if w not in ENGLISH_STOP_WORDS]
    text = [each for each in text if len(each) > 1]
    text = [each for each in text if ' ' not in each]
    return text

def get_synonyms(word):
    """Find synonyms for a given word."""
    synonyms = []
    
    # Get POS tag for the word
    pos_tag = nltk.pos_tag([word])[0][1]
    
    # Map the POS tag to WordNet POS name
    tag_map = {
        'JJ': wordnet.ADJ,
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'RB': wordnet.ADV,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB
    }
    
    wordnet_pos = tag_map.get(pos_tag[0:2])
    
    if wordnet_pos:
        for syn in wordnet.synsets(word, pos=wordnet_pos):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.append(lemma.name())
    
    return list(set(synonyms))

def get_antonyms(word):
    """Find antonyms for a given word using WordNet."""
    antonyms = []
    
    # Get POS tag for the word
    pos_tag = nltk.pos_tag([word])[0][1]
    
    # Map the POS tag to WordNet POS name
    tag_map = {
        'JJ': wordnet.ADJ,
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'RB': wordnet.ADV,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB
    }
    
    wordnet_pos = tag_map.get(pos_tag[0:2])
    
    if wordnet_pos:
        for syn in wordnet.synsets(word, pos=wordnet_pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    if antonym.name() != word and '_' not in antonym.name():
                        antonyms.append(antonym.name())
    
    return list(set(antonyms))

