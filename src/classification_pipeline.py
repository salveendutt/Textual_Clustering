from pipeline_orchestrator import IPipelineOrchestrator
from typing import Any, Dict, List, Optional
from classification_models import *
from noise_strategy import NoNoise
from tqdm.auto import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()  # Optional: also log to console
    ]
)

class ClassificationPipelineOrchestrator(IPipelineOrchestrator):
    def __init__(self) -> None:
        self.models = {}
        self.results = {}
    
    def add_model(self, model_type: str, name: Optional[str] = None):
        """
        Add a new topic model to the orchestrator.
        
        Args:
            model_type: Type of model ('SVM')
            config: Configuration parameters for the model
            name: Optional custom name for the model
        """
        # Create the model based on type
        if model_type == 'SVM':
            model = SVMModel()
        elif model_type == 'SVMRoberta':
            model = SVMRobertaModel()
        elif model_type == 'XGBoost':
            model = XGBoostModel()
        elif model_type == 'XGBoostRoberta':
            model = XGBoostRobertaModel()
        elif model_type == 'RandomForest':
            model = RandomForestModel()
        elif model_type == 'LightGBM':
            model = LightGBMModel()
        elif model_type == 'LightGBMRoberta':
            model = LightGBMRobertaModel()
        elif model_type == 'TARSZeroShot':
            model = TARSZeroShotModel()
        elif model_type == 'TARSFewShot':
            model = TARSFewShotModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if name is None:
            name = f"{model_type}"
        
        self.models[name] = model

    def add_models_grid(self, model_types: List[str], param_grid: Dict[str, List[Any]] = None):
        """
        Add multiple models using a grid of parameters.
        
        Args:
            model_types: List of model types ('SVM')
            param_grid: Dictionary mapping parameter names to lists of parameter values
                        This allows for creating multiple models with different configurations
                        
        Returns:
            List of names of the added models
        """
        added_model_names = []
        
        # If no param_grid provided, create a default one with empty params
        if param_grid is None:
            param_grid = {model_type: [{}] for model_type in model_types}
        
        for model_type in model_types:
            # Get parameters for this model type if available
            model_params = param_grid.get(model_type, [{}])
            
            # If model_params is not a list, convert it to a list
            if not isinstance(model_params, list):
                model_params = [model_params]
                
            # Create a model for each parameter combination
            for i, params in enumerate(model_params):
                if len(model_params) > 1:
                    # If multiple parameter sets, include index in name
                    name = f"{model_type}_{i+1}"
                else:
                    name = f"{model_type}"
                    
                self.add_model(model_type, name)
                added_model_names.append(name)
        
        return added_model_names

    def evaluate(self, documents_dict, noise_strategies=None):
        pass

    def evaluate_with_training(self, training_data, documents_dict, noise_strategies=None):
        results_df = pd.DataFrame()
    
        # Check input format
        if not isinstance(documents_dict, dict):
            documents, true_labels = documents_dict
            documents_dict = {'default': (documents, true_labels)}
        
        # Process each dataset
        for dataset_name, (documents, true_labels) in tqdm(documents_dict.items(), desc="Datasets", position=0):
            for name, model in tqdm(self.models.items(), desc="Models", position=1, leave=False):
                if noise_strategies is None:
                    noise_strategies = [NoNoise()]
                    
                for noise_strategy in tqdm(noise_strategies, desc="Noise Strategies", position=2, leave=False):
                    try:
                        noisy_documents = noise_strategy.apply(documents)
                        train_x, train_y = training_data[dataset_name]
                        model.fit_model(train_x, train_y)
                        
                        eval_results = model.evaluate(noisy_documents, true_labels)
                        
                        # Add metadata
                        eval_results['Model'] = name
                        eval_results['Dataset'] = dataset_name
                        eval_results['Noise'] = noise_strategy.__class__.__name__
                        
                        if results_df.empty:
                            results_df = pd.DataFrame([eval_results])
                        else:
                            new_df = pd.DataFrame([eval_results])
                            # Explicitly cast columns to match dtypes in results_df
                            new_df = new_df.astype({col: results_df[col].dtype for col in results_df.columns if col in new_df.columns})
                            results_df = pd.concat([results_df, new_df], ignore_index=True)
                    except Exception as e:
                        print(f"Error evaluating model {name}: {str(e)}")
        
        # Store and return results - updated to use classification metrics
        self.results = results_df[['Dataset', 'Noise', 'Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall']].sort_values(by=['Dataset', 'Noise', 'Model'])
        return self.results

        