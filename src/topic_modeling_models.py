import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import (LatentDirichletAllocation, NMF, TruncatedSVD)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer)
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from tools import process_text

# Interface for Topic Models
class ITopicModel(ABC):
    @abstractmethod
    def __init__(self):
        self.assigned_topics = None
        self.topic_distributions = None

    @abstractmethod
    def fit_transform(self, documents):
        pass
    
    @abstractmethod
    def get_topics(self, n_words=10):
        pass
    
    def assign_topics(self, documents_processed: list):
        topic_distributions = self.fit_transform(documents_processed)
        assigned_topics = topic_distributions.argmax(axis=1)
        self.assigned_topics = assigned_topics
        self.topic_distributions = topic_distributions
    
    def evaluate(self, documents, true_labels: pd.DataFrame=None):
        documents_processed = documents.apply(process_text).tolist()
        true_labels = true_labels.tolist()

        self.assign_topics(documents_processed)

        topic_distributions = self.topic_distributions
        assigned_topics = self.assigned_topics
        
        # ARI Score
        ari_score = adjusted_rand_score(true_labels, assigned_topics)
        
        # Topic Coherence
        tokenized_docs = documents_processed
        dictionary = Dictionary(tokenized_docs)
        topic_words = self.get_topics()
        
        coherence_model = CoherenceModel(
            topics=topic_words, 
            texts=tokenized_docs, 
            dictionary=dictionary, 
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Cosine Similarity
        cosine_sim = np.mean(cosine_similarity(topic_distributions))
        
        # Reconstruction Error (only for NMF)
        reconstruction_error = self.model.reconstruction_err_ if isinstance(self, NMFModel) else None
        
        return {
            "ARI Score": ari_score,
            "Topics Coherence": coherence_score,
            "Cosine Similarity": cosine_sim,
            "Reconstruction Error": reconstruction_error
        }


# LDA Implementation
class LDAModel(ITopicModel):
    def __init__(self, n_topics=5):
        super().__init__()
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(stop_words='english', preprocessor=' '.join)
        self.model = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    def fit_transform(self, documents):
        self.term_matrix = self.vectorizer.fit_transform(documents)
        return self.model.fit_transform(self.term_matrix)

    def get_topics(self, n_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
        return topics

# LSI Implementation
class LSIModel(ITopicModel):
    def __init__(self, n_topics=5):
        super().__init__()
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        self.model = TruncatedSVD(n_components=n_topics, random_state=42)

    def fit_transform(self, documents):
        self.term_matrix = self.vectorizer.fit_transform(documents)
        return self.model.fit_transform(self.term_matrix)

    def get_topics(self, n_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
        return topics

# NMF Implementation
class NMFModel(ITopicModel):
    def __init__(self, n_topics=5):
        super().__init__()
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        self.model = NMF(n_components=n_topics, random_state=42)

    def fit_transform(self, documents):
        self.term_matrix = self.vectorizer.fit_transform(documents)
        return self.model.fit_transform(self.term_matrix)

    def get_topics(self, n_words=5):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]])
        return topics