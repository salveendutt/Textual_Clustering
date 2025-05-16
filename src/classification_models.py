import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from tools import process_text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ISupervisedClasificationModel(ABC):
    @abstractmethod
    def __init__(self):
        self.assigned_topics = None
        self.topic_distributions = None

    # @abstractmethod
    # def fit_model(self, documents):
    #     pass

    @abstractmethod
    def predict_classes(self, documents): 
        pass

    @abstractmethod    
    def fit_model(self, train_data_X, train_data_y):
        pass

    def evaluate(self, documents, true_labels: pd.DataFrame=None):
        documents_processed = documents.apply(process_text).tolist()
        accuracy = accuracy_score(self.predict_classes(documents_processed), true_labels)
        f1 = f1_score(self.predict_classes(documents_processed), true_labels, average='weighted')
        precision = precision_score(self.predict_classes(documents_processed), true_labels, average='weighted')
        recall = recall_score(self.predict_classes(documents_processed), true_labels, average='weighted')
        return {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        }


class SVMModel(ISupervisedClasificationModel):
    def __init__(self):
        super().__init__()
        self.vectorizer = CountVectorizer(stop_words='english', preprocessor=' '.join)
        self.model = SVC()
    
    def fit_model(self, train_data_X, train_data_y):
        self.term_matrix = self.vectorizer.fit_transform(train_data_X)
        return self.model.fit(self.term_matrix, train_data_y)
    
    def predict_classes(self, documents):
        return self.model.predict(self.vectorizer.fit_transform(documents))
    

