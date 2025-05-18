import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tools import process_text
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import RobertaModel, RobertaTokenizer
import torch


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
        predictions = self.predict_classes(documents)
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
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        self.model = SVC()
    
    def fit_model(self, train_data_X, train_data_y):
        documents_processed = train_data_X.apply(process_text).tolist()
        self.term_matrix = self.vectorizer.fit_transform(documents_processed)
        return self.model.fit(self.term_matrix, train_data_y)
    
    def predict_classes(self, documents):
        documents_processed = documents.apply(process_text).tolist()
        return self.model.predict(self.vectorizer.transform(documents_processed))
    

class SVMRobertaModel(IClasificationModel):
    def __init__(self, model_name='distilroberta-base', max_length=1024, batch_size=16):
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
        self.model = LinearSVC(C=1.0, max_iter=10000)
        
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
        documents_processed = train_data_X.tolist()
        self.embeddings = self._get_roberta_embeddings(documents_processed)
        return self.model.fit(self.embeddings, train_data_y)
    
    def predict_classes(self, documents):
        documents_processed = documents.tolist()
        embeddings = self._get_roberta_embeddings(documents_processed)
        return self.model.predict(embeddings)


class XGBoostModel(IClasificationModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        super().__init__()
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric='mlogloss'  # Removed use_label_encoder parameter
        )
        self.label_encoder = None
        self.original_classes = None
        
    def fit_model(self, train_data_X, train_data_y):
        documents_processed = train_data_X.apply(process_text).tolist()
        self.term_matrix = self.vectorizer.fit_transform(documents_processed)
        
        # Handle non-zero based labels (e.g., [1,2,3,4] instead of [0,1,2,3])
        self.original_classes = np.unique(train_data_y)
        if np.min(self.original_classes) > 0:
            # Create mapping from original labels to zero-based indices
            self.label_encoder = {original: idx for idx, original in enumerate(self.original_classes)}
            # Transform labels to zero-based
            transformed_y = np.array([self.label_encoder[label] for label in train_data_y])
            return self.model.fit(self.term_matrix, transformed_y)
        else:
            return self.model.fit(self.term_matrix, train_data_y)
    
    def predict_classes(self, documents):
        documents_processed = documents.apply(process_text).tolist()
        predictions = self.model.predict(self.vectorizer.transform(documents_processed))
        
        # If we transformed the labels for training, convert predictions back to original labels
        if self.label_encoder is not None:
            # Create reverse mapping from indices to original labels
            reverse_encoder = {idx: original for original, idx in self.label_encoder.items()}
            return np.array([reverse_encoder[pred] for pred in predictions])
        else:
            return predictions


class RandomForestModel(IClasificationModel):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__()
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
    def fit_model(self, train_data_X, train_data_y):
        documents_processed = train_data_X.apply(process_text).tolist()
        self.term_matrix = self.vectorizer.fit_transform(documents_processed)
        return self.model.fit(self.term_matrix, train_data_y)
    
    def predict_classes(self, documents):
        documents_processed = documents.apply(process_text).tolist()
        return self.model.predict(self.vectorizer.transform(documents_processed))


class LightGBMModel(IClasificationModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, min_child_samples=20, 
                 num_leaves=31, min_data_in_leaf=20, use_tfidf=True, verbose=-1):
        super().__init__()
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            random_state=42,
            verbose=verbose,
            # Parameters to help with sparse data
            min_gain_to_split=0.0,
            boosting_type='gbdt'
        )
        self.label_encoder = None
        self.original_classes = None
        
    def fit_model(self, train_data_X, train_data_y):
        documents_processed = train_data_X.apply(process_text).tolist()
        
        # For text data, TF-IDF often works better
        self.term_matrix = self.vectorizer.fit_transform(documents_processed)
        
        # Handle non-zero based labels as we did with XGBoost
        self.original_classes = np.unique(train_data_y)
        if np.min(self.original_classes) > 0:
            # Create mapping from original labels to zero-based indices
            self.label_encoder = {original: idx for idx, original in enumerate(self.original_classes)}
            # Transform labels to zero-based
            transformed_y = np.array([self.label_encoder[label] for label in train_data_y])
            
            # LightGBM works better with dense arrays for sparse data
            return self.model.fit(
                self.term_matrix.toarray(), 
                transformed_y,
                # Add categorical feature info explicitly
                categorical_feature='auto'
            )
        else:
            return self.model.fit(
                self.term_matrix.toarray(), 
                train_data_y,
                categorical_feature='auto'
            )
    
    def predict_classes(self, documents):
        documents_processed = documents.apply(process_text).tolist()
        predictions = self.model.predict(self.vectorizer.transform(documents_processed).toarray())
        
        # If we transformed the labels for training, convert predictions back to original labels
        if self.label_encoder is not None:
            # Create reverse mapping from indices to original labels
            reverse_encoder = {idx: original for original, idx in self.label_encoder.items()}
            return np.array([reverse_encoder[pred] for pred in predictions])
        else:
            return predictions


class XGBoostRobertaModel(IClasificationModel):
    def __init__(self,
                 model_name='distilroberta-base',
                 max_length=None,
                 batch_size=16,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=5):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'mps' if torch.backends.mps.is_available()
                                   else 'cpu')

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.roberta_model = RobertaModel.from_pretrained(self.model_name).to(self.device)

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        self.label_encoder = None
        self.original_classes = None

    def _get_roberta_embeddings(self, texts):
        embeddings = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                out = self.roberta_model(**encoded)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(emb)
        return np.array(embeddings)

    def fit_model(self, train_data_X, train_data_y):
        texts = train_data_X.tolist()
        X_emb = self._get_roberta_embeddings(texts)

        self.original_classes = np.unique(train_data_y)
        if np.min(self.original_classes) > 0:
            self.label_encoder = {c: i for i, c in enumerate(self.original_classes)}
            y = np.array([self.label_encoder[c] for c in train_data_y])
        else:
            y = train_data_y.values

        return self.model.fit(X_emb, y)

    def predict_classes(self, documents):
        texts = documents.tolist()
        X_emb = self._get_roberta_embeddings(texts)
        preds = self.model.predict(X_emb)

        if self.label_encoder is not None:
            inv = {v: k for k, v in self.label_encoder.items()}
            return np.array([inv[p] for p in preds])
        return preds


class LightGBMRobertaModel(IClasificationModel):
    def __init__(self,
                 model_name='distilroberta-base',
                 max_length=None,
                 batch_size=16,
                 n_estimators=100,
                 learning_rate=0.1,
                 max_depth=-1,
                 num_leaves=31,
                 min_child_samples=20):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'mps' if torch.backends.mps.is_available()
                                   else 'cpu')

        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.roberta_model = RobertaModel.from_pretrained(self.model_name).to(self.device)

        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            random_state=42,
            boosting_type='gbdt',
            force_col_wise=True,
        )
        self.label_encoder = None
        self.original_classes = None

    def _get_roberta_embeddings(self, texts):
        embeddings = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                out = self.roberta_model(**encoded)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(emb)
        return np.array(embeddings)

    def fit_model(self, train_data_X, train_data_y):
        texts = train_data_X.tolist()
        X_emb = self._get_roberta_embeddings(texts)

        self.original_classes = np.unique(train_data_y)
        if np.min(self.original_classes) > 0:
            self.label_encoder = {c: i for i, c in enumerate(self.original_classes)}
            y = np.array([self.label_encoder[c] for c in train_data_y])
        else:
            y = train_data_y.values

        return self.model.fit(X_emb, y)

    def predict_classes(self, documents):
        texts = documents.tolist()
        X_emb = self._get_roberta_embeddings(texts)
        preds = self.model.predict(X_emb)

        if self.label_encoder is not None:
            inv = {v: k for k, v in self.label_encoder.items()}
            return np.array([inv[p] for p in preds])
        return preds
