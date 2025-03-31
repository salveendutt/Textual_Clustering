from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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