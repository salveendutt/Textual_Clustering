import re
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.casual import casual_tokenize
from sklearn.decomposition import (
    LatentDirichletAllocation,
    NMF,
    TruncatedSVD,
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


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


# Interface for Topic Models
class ITopicModel(ABC):
    @abstractmethod
    def fit_transform(self, documents):
        pass
    
    @abstractmethod
    def get_topics(self, n_words=10):
        pass


# LDA Implementation
class LDAModel(ITopicModel):
    def __init__(self, n_topics=5):
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
        

# Pipeline for running any topic model
class TopicModelPipeline:
    def __init__(self, model: ITopicModel):
        self.model = model
        self.assigned_topics = None
        self.topic_distributions = None

    def get_topics(self):
        return self.model.get_topics()

    def assign_topics(self, documents_processed: list):
        topic_distributions = self.model.fit_transform(documents_processed)
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
        reconstruction_error = self.model.model.reconstruction_err_ if isinstance(self.model, NMFModel) else None
        
        return {
            "ARI Score": ari_score,
            "Topic Coherence": coherence_score,
            "Cosine Similarity": cosine_sim,
            "Reconstruction Error": reconstruction_error
        }


class TopicModelOrchestrator:
    """
    A class to orchestrate multiple topic model pipelines with different configurations
    and evaluate their performance.
    """
    def __init__(self):
        self.pipelines = {}
        self.results = {}
        
    def add_model(self, model_type: str, config: Dict[str, Any], name: Optional[str] = None):
        """
        Add a new topic model to the orchestrator.
        
        Args:
            model_type: Type of model ('LDA', 'LSI', or 'NMF')
            config: Configuration parameters for the model
            name: Optional custom name for the model
        """
        if model_type not in ['LDA', 'LSI', 'NMF']:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'LDA', 'LSI', or 'NMF'")
        
        # Create the model based on type
        if model_type == 'LDA':
            model = LDAModel(**config)
        elif model_type == 'LSI':
            model = LSIModel(**config)
        else:  # NMF
            model = NMFModel(**config)
        
        # Create a pipeline for the model
        pipeline = TopicModelPipeline(model)
        
        # Generate a name if not provided
        if name is None:
            n_topics = config.get('n_topics', 5)
            name = f"{model_type}_{n_topics}"
            
        # Add counter suffix if name already exists
        base_name = name
        counter = 1
        while name in self.pipelines:
            name = f"{base_name}_{counter}"
            counter += 1
            
        self.pipelines[name] = pipeline
        return name
    
    def add_models_grid(self, model_types: List[str], param_grid: Dict[str, List[Any]]):
        """
        Add multiple models using a grid of parameters.
        
        Args:
            model_types: List of model types ('LDA', 'LSI', 'NMF')
            param_grid: Dictionary of parameter lists to create a grid
                        Example: {'n_topics': [5, 10, 15]}
        
        Returns:
            List of names of the added models
        """
        added_models = []
        
        # Create all combinations of parameters
        import itertools
        param_names = param_grid.keys()
        param_values = [param_grid[name] for name in param_names]
        
        for model_type in model_types:
            for params in itertools.product(*param_values):
                config = {name: value for name, value in zip(param_names, params)}
                name = self.add_model(model_type, config)
                added_models.append(name)
                
        return added_models
    
    def evaluate(self, documents_dict, sort_by=None):
        """
        Evaluate all models in the orchestrator on multiple datasets and store results.
        
        Args:
            documents_dict: Dictionary where keys are dataset names and values are tuples of (documents, true_labels)
                        Each documents should be a pandas Series/list of text documents
                        true_labels can be None if not available
            sort_by: Optional metric to sort results by
                
        Returns:
            Dictionary of DataFrames with results for each dataset
        """
        all_results = {}
        
        # Check input format
        if not isinstance(documents_dict, dict):
            # Backward compatibility - assume it's a single dataset
            documents = documents_dict
            true_labels = sort_by  # In the old interface, this was the second parameter
            sort_by = None
            documents_dict = {'default': (documents, true_labels)}
        
        # Process each dataset
        for dataset_name, (documents, true_labels) in documents_dict.items():
            results = {}
            
            print(f"Evaluating models on dataset: {dataset_name}")
            
            for name, pipeline in self.pipelines.items():
                try:
                    eval_results = pipeline.evaluate(documents, true_labels)
                    # Add model name and dataset to results
                    eval_results['Model'] = name
                    eval_results['Dataset'] = dataset_name
                    results[name] = eval_results
                    print(f"  Evaluated model: {name}")
                except Exception as e:
                    print(f"  Error evaluating model {name}: {str(e)}")
                    
            # Convert results to DataFrame
            if results:
                results_df = pd.DataFrame.from_dict(results, orient='index')
                
                # Sort results if requested
                if sort_by is not None and sort_by in results_df.columns:
                    # For coherence and ARI, higher is better
                    if sort_by in ['Topic Coherence', 'ARI Score']:
                        results_df = results_df.sort_values(by=sort_by, ascending=False)
                    # For cosine similarity and reconstruction error, lower is better
                    else:
                        results_df = results_df.sort_values(by=sort_by, ascending=True)
                        
                all_results[dataset_name] = results_df
            else:
                all_results[dataset_name] = pd.DataFrame()

        merged_results = pd.concat(all_results.values(), keys=all_results.keys(), names=['Dataset'])
        # Store all results in instance
        self.results = merged_results
                
        return merged_results

        
    def print_topics(self, model_name=None, n_words=10):
        """
        Print the topics discovered by a model.
        
        Args:
            model_name: Name of the model (if None, prints for all models)
            n_words: Number of words to show for each topic
        """
        if model_name is not None:
            if model_name not in self.pipelines:
                raise ValueError(f"Unknown model: {model_name}")
                
            pipeline = self.pipelines[model_name]
            topics = pipeline.get_topics()
            
            print(f"Topics for model {model_name}:")
            for i, topic in enumerate(topics):
                print(f"Topic {i}: {', '.join(topic[:n_words])}")
            print()
        else:
            for name, pipeline in self.pipelines.items():
                topics = pipeline.get_topics()
                
                print(f"Topics for model {name}:")
                for i, topic in enumerate(topics):
                    print(f"Topic {i}: {', '.join(topic[:n_words])}")
                print()