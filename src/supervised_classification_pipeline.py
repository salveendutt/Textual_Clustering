from pipeline_orchestrator import IPipelineOrchestrator
from typing import Any, Dict, List, Optional
from supervised_classification_models import *

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
