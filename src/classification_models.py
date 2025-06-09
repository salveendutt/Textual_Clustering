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
from flair.models import TARSClassifier
from flair.data import Sentence
from typing import List, Optional, Union
from langchain_ollama import OllamaLLM
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.getLogger("httpx").setLevel(logging.WARNING)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm.auto import tqdm


class IClasificationModel(ABC):
    @abstractmethod
    def __init__(self):
        self.assigned_topics = None

    @abstractmethod
    def predict_classes(self, documents): 
        pass

    @abstractmethod    
    def fit_model(self, train_data_X, train_data_y):
        pass

    def evaluate(self, documents, true_labels: pd.DataFrame=None):
        predictions = self.predict_classes(documents)
        df = pd.DataFrame({'Documents': documents, 'Predictions': predictions, 'True Labels': true_labels})
        accuracy = accuracy_score(predictions, true_labels)
        f1 = f1_score(predictions, true_labels, average='weighted')
        precision = precision_score(predictions, true_labels, average='weighted')
        recall = recall_score(predictions, true_labels, average='weighted')

        return {
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        }, df


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
    def __init__(self, model_name='distilroberta-base', max_length=None, batch_size=8):
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


class RandomForestRobertaModel(IClasificationModel):
    def __init__(self, 
                 model_name='distilroberta-base', 
                 max_length=None, 
                 batch_size=16,
                 n_estimators=100, 
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=42):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Device selection following the pattern from other RoBERTa models
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                   else 'mps' if torch.backends.mps.is_available()
                                   else 'cpu')
        
        # Load RoBERTa tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.roberta_model = RobertaModel.from_pretrained(self.model_name).to(self.device)
        
        # Initialize Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        # Label encoding for non-zero based labels
        self.label_encoder = None
        self.original_classes = None
    
    def _get_roberta_embeddings(self, texts):
        """
        Generate embeddings for texts using RoBERTa
        """
        embeddings = []
        
        for text in texts:
            # Tokenize a single text
            encoded_input = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.roberta_model(**encoded_input)
            
            # Use the [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def fit_model(self, train_data_X, train_data_y):
        # Convert to list if pandas Series
        texts = train_data_X.tolist()
        
        # Get RoBERTa embeddings
        X_embeddings = self._get_roberta_embeddings(texts)
        
        # Handle non-zero based labels (following the pattern from XGBoost and LightGBM models)
        self.original_classes = np.unique(train_data_y)
        if np.min(self.original_classes) > 0:
            # Create mapping from original labels to zero-based indices
            self.label_encoder = {original: idx for idx, original in enumerate(self.original_classes)}
            # Transform labels to zero-based
            transformed_y = np.array([self.label_encoder[label] for label in train_data_y])
            return self.model.fit(X_embeddings, transformed_y)
        else:
            # Labels are already zero-based
            y = train_data_y.values if hasattr(train_data_y, 'values') else train_data_y
            return self.model.fit(X_embeddings, y)
    
    def predict_classes(self, documents):
        # Convert to list if pandas Series
        texts = documents.tolist()
        
        # Get RoBERTa embeddings
        X_embeddings = self._get_roberta_embeddings(texts)
        
        # Make predictions
        predictions = self.model.predict(X_embeddings)
        
        # If we transformed the labels for training, convert predictions back to original labels
        if self.label_encoder is not None:
            # Create reverse mapping from indices to original labels
            reverse_encoder = {idx: original for original, idx in self.label_encoder.items()}
            return np.array([reverse_encoder[pred] for pred in predictions])
        else:
            return predictions
    
    def predict_proba(self, documents):
        """
        Get prediction probabilities (useful for Random Forest)
        """
        texts = documents.tolist()
        X_embeddings = self._get_roberta_embeddings(texts)
        return self.model.predict_proba(X_embeddings)
    
    def get_feature_importance(self):
        """
        Get feature importance from the Random Forest model
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            raise ValueError("Model must be fitted before getting feature importance")


class LightGBMModel(IClasificationModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5, min_child_samples=20, 
                 num_leaves=31, min_data_in_leaf=20, use_tfidf=True, verbose=-1):
        super().__init__()
        self.vectorizer = TfidfVectorizer(stop_words='english', preprocessor=' '.join)
        
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            # min_child_samples=min_child_samples,
            # num_leaves=num_leaves,
            # min_data_in_leaf=min_data_in_leaf,
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
                self.term_matrix,
                transformed_y,
                # Add categorical feature info explicitly
                categorical_feature='auto'
            )
        else:
            return self.model.fit(
                self.term_matrix,
                train_data_y,
                categorical_feature='auto'
            )
    
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

class TARSZeroShotModel(IClasificationModel):
    def __init__(self, model_name: str = 'tars-base'):
        super().__init__()
        self.model = TARSClassifier.load(model_name)
        
        self.topics = None
        self.topics_mapping = None

    def fit_model(self, train_data_X, train_data_y):
        pass

    def _generate_unique_task_name(self):
        """Generate a unique task name based on current topics
        If this step is ommited then the model will not use new topics and reuse the old ones."""
        # Option 1: Use UUID for completely unique names
        return f"ZeroShot_{uuid.uuid4().hex[:8]}"

    def set_topics(self, topics: List[str]):
        """
        Set the topics for classification.
        
        Args:
            topics: List of topic names to classify the documents into.
        """
        self.topics = topics

    def set_topics_mapping(self, mapping: List[str]):
        """
        Set the topics for classification.
        
        Args:
            topics: List of topic names to classify the documents into.
        """
        self.topics_mapping = mapping

    def predict_classes(self, documents):
            
        predictions = []
        
        self.model.add_and_switch_to_new_task(self._generate_unique_task_name(), label_dictionary=self.topics, label_type="classification")
        for doc in documents.tolist():
            sentence = Sentence(doc)
            self.model.predict(sentence)
            if sentence.labels:
                prediction = sentence.labels[0].value
            else:
                prediction = "Unknown"
                
            predictions.append(prediction)
        preds_mapped = [self.topics_mapping.get(pred, 0) for pred in predictions]
        return np.array(preds_mapped)

class LLMClassifierModel(IClasificationModel):
    def __init__(self, model_name: str = 'gemma3'):
        self.model = OllamaLLM(model=model_name)
        self.topics = None
        self.topics_mapping = None
    def fit_model(self, train_data_X, train_data_y):
        pass
    
    def set_topics(self, topics: List[str]):
        """
        Set the topics for classification.
        
        Args:
            topics: List of topic names to classify the documents into.
        """
        self.topics = topics

    def set_topics_mapping(self, mapping: List[str]):
        """
        Set the topics for classification.
        
        Args:
            topics: List of topic names to classify the documents into.
        """
        self.topics_mapping = mapping

    def generate_prompt(self, document):

        return f'''You are provided with news and helping to classify them based on the topics.
Please assign the news to the topics provided. Return only the name of the topic for the respective news.
News can be found in tripletick block: ```{document}```
Topics to choose from: {self.topics}
Please return ONLY the topic name. DO NOT OUTPUT any additional text, quotes, or formatting.'''

    def predict_classes(self, documents):
            
        predictions = []
        
        for doc in tqdm(documents.tolist(), desc="Classifying with LLM"):
            try:
                prompt = self.generate_prompt(doc)
                response = self.model.invoke(prompt)
                response = response.strip()
                prediction = response.replace('``', '').replace('"', '').replace("'", "")
                predictions.append(prediction)
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                predictions.append("Unknown")
        text_to_class_mapping = self.topics_mapping
        preds_mapped = [text_to_class_mapping.get(pred, -1) for pred in predictions]
        return np.array(preds_mapped)

# class LLMClassifierModel(IClasificationModel):
#     def __init__(self, model_name: str = 'gemma3'):
#         self.model = OllamaLLM(model=model_name)
#         self.topics = None
#         self.topics_mapping = None
#         self._lock = threading.Lock()  # For thread-safe operations if needed
        
#     def fit_model(self, train_data_X, train_data_y):
#         pass
        
#     def set_topics(self, topics: List[str]):
#         """
#         Set the topics for classification.
#         Args:
#             topics: List of topic names to classify the documents into.
#         """
#         self.topics = topics
        
#     def set_topics_mapping(self, mapping: List[str]):
#         """
#         Set the topics for classification.
#         Args:
#             topics: List of topic names to classify the documents into.
#         """
#         self.topics_mapping = mapping
        
#     def generate_prompt(self, document):
#         return f'''You are provided with news and helping to classify them based on the topics.
# Please assign the news to the topics provided. Return only the name of the topic for the respective news.
# News can be found in tripletick block: ```{document}```
# Topics to choose from: {self.topics}
# Please return ONLY the topic name. DO NOT OUTPUT any additional text, quotes, or formatting.'''

#     def _classify_single_document(self, doc_with_index):
#         """
#         Classify a single document. This method will be called by each thread.
#         Args:
#             doc_with_index: Tuple of (index, document) to maintain order
#         Returns:
#             Tuple of (index, prediction)
#         """
#         index, doc = doc_with_index
#         try:
#             prompt = self.generate_prompt(doc)
#             response = self.model.invoke(prompt)
#             response = response.strip()
#             prediction = response.replace('``', '').replace('"', '').replace("'", "")
#             return index, prediction
#         except Exception as e:
#             print(f"Error processing document at index {index}: {str(e)}")
#             return index, ""  # Return empty string for failed predictions

#     def predict_classes(self, documents):
#         """
#         Predict classes for documents using multithreading with max 4 concurrent threads.
#         Args:
#             documents: Array-like of documents to classify
#         Returns:
#             numpy array of predicted class indices
#         """
#         docs_list = documents.tolist()
#         predictions = [None] * len(docs_list)  # Pre-allocate list to maintain order
        
#         # Create list of (index, document) tuples to maintain order
#         indexed_docs = list(enumerate(docs_list))
        
#         # Use ThreadPoolExecutor with max 4 workers
#         with ThreadPoolExecutor(max_workers=4, thread_name_prefix="LLM-Classifier") as executor:
#             # Submit all tasks
#             future_to_index = {
#                 executor.submit(self._classify_single_document, doc_with_index): doc_with_index[0] 
#                 for doc_with_index in indexed_docs
#             }
            
#             # Process completed tasks with progress bar
#             with tqdm(total=len(docs_list), desc="Classifying with LLM (4 threads)") as pbar:
#                 for future in as_completed(future_to_index):
#                     try:
#                         index, prediction = future.result()
#                         predictions[index] = prediction
#                         pbar.update(1)
#                     except Exception as e:
#                         index = future_to_index[future]
#                         print(f"Thread failed for document {index}: {str(e)}")
#                         predictions[index] = ""  # Default for failed predictions
#                         pbar.update(1)
        
#         # Map text predictions to class indices
#         text_to_class_mapping = self.topics_mapping
#         preds_mapped = [text_to_class_mapping.get(pred, -1) for pred in predictions]
        
#         return np.array(preds_mapped)

#     def predict_classes_with_detailed_progress(self, documents):
#         """
#         Alternative method with more detailed progress tracking.
#         Shows both submitted and completed tasks.
#         """
#         docs_list = documents.tolist()
#         predictions = [None] * len(docs_list)
#         indexed_docs = list(enumerate(docs_list))
        
#         with ThreadPoolExecutor(max_workers=4, thread_name_prefix="LLM-Classifier") as executor:
#             # Submit all tasks and track futures
#             futures = []
#             print(f"Submitting {len(indexed_docs)} classification tasks...")
            
#             for doc_with_index in indexed_docs:
#                 future = executor.submit(self._classify_single_document, doc_with_index)
#                 futures.append((future, doc_with_index[0]))
            
#             print(f"All tasks submitted. Processing with max 4 concurrent threads...")
            
#             # Process results as they complete
#             completed = 0
#             with tqdm(total=len(docs_list), desc="Classifying") as pbar:
#                 for future, original_index in futures:
#                     try:
#                         index, prediction = future.result()
#                         predictions[index] = prediction
#                         completed += 1
#                         pbar.set_postfix({
#                             'completed': completed, 
#                             'active_threads': min(4, len(docs_list) - completed)
#                         })
#                         pbar.update(1)
#                     except Exception as e:
#                         print(f"Error processing document {original_index}: {str(e)}")
#                         predictions[original_index] = ""
#                         pbar.update(1)
        
#         # Map predictions to class indices
#         text_to_class_mapping = self.topics_mapping
#         preds_mapped = [text_to_class_mapping.get(pred, -1) for pred in predictions]
        
#         return np.array(preds_mapped)