import os
from .config import MODEL_PATHS

class ModelEvaluator:
    """Headless model evaluation utilities"""
    
    @staticmethod
    def get_available_evaluations():
        """Get available model evaluation types - headless"""
        available = []
        for eval_type, path in MODEL_PATHS.items():
            if os.path.exists(path) and any(f for f in os.listdir(path) 
                                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
                available.append(eval_type)
        return available
    
    @staticmethod
    def get_evaluation_images(evaluation_type):
        """Get images for specific evaluation type - headless"""
        from .preprocessing import ImageManager
        return ImageManager.get_model_evaluation_images(evaluation_type)