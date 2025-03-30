from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
from topic_modeling_models import *

class IPipelineOrchestrator(ABC):
    @abstractmethod
    def add_model(self, model_type: str, config: Dict[str, Any], name: Optional[str] = None):
        pass

    @abstractmethod
    def add_models_grid(self, model_types: List[str], param_grid: Dict[str, List[Any]]):
        pass

    @abstractmethod
    def evaluate(self, documents_dict, sort_by=None):
        pass

    # @abstractmethod
    # def print_topics(self, model_name=None, n_words=10):
    #     pass

class TopicModelingPipelineOrchestrator(IPipelineOrchestrator):
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
        elif model_type == 'NMF':
            model = NMFModel(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
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
    
    def evaluate(self, documents_dict, noise_strategies=None):
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
        # results_df = pd.DataFrame()
        all_results = {}
        
        # Check input format
        if not isinstance(documents_dict, dict):
            # assume it's a single dataset
            documents = documents_dict
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
                except Exception as e:
                    print(f"  Error evaluating model {name}: {str(e)}")
                    
            # Convert results to DataFrame
            if results:
                results_df = pd.DataFrame.from_dict(results, orient='index')                  
                all_results[dataset_name] = results_df
            else:
                all_results[dataset_name] = pd.DataFrame()

        merged_results = pd.concat(all_results.values(), keys=all_results.keys())

        self.results = merged_results.reset_index(level=0, drop=True).reset_index()
                
        return

        
    # def print_topics(self, model_name=None, n_words=10):
    #     """
    #     Print the topics discovered by a model.
        
    #     Args:
    #         model_name: Name of the model (if None, prints for all models)
    #         n_words: Number of words to show for each topic
    #     """
    #     if model_name is not None:
    #         if model_name not in self.pipelines:
    #             raise ValueError(f"Unknown model: {model_name}")
                
    #         pipeline = self.pipelines[model_name]
    #         topics = pipeline.get_topics()
            
    #         print(f"Topics for model {model_name}:")
    #         for i, topic in enumerate(topics):
    #             print(f"Topic {i}: {', '.join(topic[:n_words])}")
    #         print()
    #     else:
    #         for name, pipeline in self.pipelines.items():
    #             topics = pipeline.get_topics()
                
    #             print(f"Topics for model {name}:")
    #             for i, topic in enumerate(topics):
    #                 print(f"Topic {i}: {', '.join(topic[:n_words])}")
    #             print()