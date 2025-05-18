import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from tools import process_text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaModel, RobertaTokenizer
import torch

import logging


class IClasificationModel(ABC):
    @abstractmethod
    def __init__(self):
        self.assigned_topics = None
        self.topic_distributions = None

    @abstractmethod
    def predict_classes(self, documents): 
        pass

    @abstractmethod    
    def fit_model(self, train_data_X, train_data_y):
        pass

    def evaluate(self, documents, true_labels: pd.DataFrame=None):
        documents_processed = documents.apply(process_text).tolist()
        predictions = self.predict_classes(documents_processed)
        accuracy = accuracy_score(predictions, true_labels)
        f1 = f1_score(predictions, true_labels, average='weighted')
        precision = precision_score(predictions, true_labels, average='weighted')
        recall = recall_score(predictions, true_labels, average='weighted')

        return {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        }


class SVMModel(IClasificationModel):
    def __init__(self):
        super().__init__()
        self.vectorizer = CountVectorizer(stop_words='english', preprocessor=' '.join)
        self.model = SVC()
    
    def fit_model(self, train_data_X, train_data_y):
        documents_processed = train_data_X.apply(process_text).tolist()
        self.term_matrix = self.vectorizer.fit_transform(documents_processed)
        return self.model.fit(self.term_matrix, train_data_y)
    
    def predict_classes(self, documents):
        return self.model.predict(self.vectorizer.transform(documents))
    
class SVMRobertaModel(IClasificationModel):
    def __init__(self, model_name='distilroberta-base', max_length=512, batch_size=16):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device('mps')
        
        # Load RoBERTa tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.roberta_model = RobertaModel.from_pretrained(self.model_name).to(self.device)
        
        # Initialize SVM classifier
        self.model = SVC()
        
    def _get_roberta_embeddings(self, texts):
        """
        Generate embeddings for texts using RoBERTa without batch processing
        """
        embeddings = []

        for text in texts:
            # Tokenize a single text
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.roberta_model(**encoded_input)

            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(embedding)

        return np.array(embeddings)
    
    def fit_model(self, train_data_X, train_data_y):
        documents_processed = train_data_X.apply(process_text).tolist()
        logger = logging.getLogger(__name__)
        logger.info(f"Processing {len(documents_processed)} documents for RoBERTa embeddings.")
        self.embeddings = self._get_roberta_embeddings(documents_processed)
        return self.model.fit(self.embeddings, train_data_y)
    
    def predict_classes(self, documents):
        embeddings = self._get_roberta_embeddings(documents)
        return self.model.predict(embeddings)
