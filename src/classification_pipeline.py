from pipeline_orchestrator import IPipelineOrchestrator
from typing import Any, Dict, List, Optional
from classification_models import *
from noise_strategy import NoNoise
import tqdm

class SupervisedClassificationPipelineOrchestrator(IPipelineOrchestrator):
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
        if model_type not in ['SVM']:
            raise ValueError(f"Unknown model type: {model_type}. Expected 'SVM'")
        
        # Create the model based on type
        if model_type == 'SVM':
            model = SVMModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if name is None:
            name = f"{model_type}"
        
        self.models[name] = model

    def add_models_grid(self, model_types: List[str]):
        """
        Add multiple models using a grid of parameters.
        
        Args:
            model_types: List of model types ('SVM')
        
        Returns:
            List of names of the added models
        """
        
        for model_type in model_types:
                name = self.add_model(model_type)

    def evaluate(self, documents_dict, noise_strategies=None):
        results_df = pd.DataFrame()
        # Check input format
        if not isinstance(documents_dict, dict):
            documents = documents_dict
            documents_dict = {'default': (documents, true_labels)}
        
        # Process each dataset
        for dataset_name, (documents, true_labels) in tqdm(documents_dict.items(), desc="Datasets", position=0):
            for name, model in tqdm(self.models.items(), desc="Models", position=1, leave=False):
                if noise_strategies == None:
                    noise_strategies = [NoNoise()]

                for noise_strategy in tqdm(noise_strategies, desc="Noise Strategies", position=2, leave=False):
                    try:
                        noisy_documents = noise_strategy.apply(documents)
                        eval_results = model.evaluate(noisy_documents, true_labels)
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
                        print(f"  Error evaluating model {name}: {str(e)}")
                    
        self.results = results_df[['Dataset', 'Noise', 'Model', 'ARI Score', 'Topics Coherence', 'Cosine Similarity', 'Reconstruction Error']].sort_values(by=['Dataset', 'Noise', 'Model'])
        return

        