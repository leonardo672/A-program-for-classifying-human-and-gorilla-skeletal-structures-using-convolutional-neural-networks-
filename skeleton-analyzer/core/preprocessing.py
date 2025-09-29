import os
import cv2
import numpy as np
from .config import *

class ImageManager:
    """Headless image management"""
    
    @staticmethod
    def get_images_from_folder(folder_path):
        """Get all images from folder - headless"""
        if not os.path.exists(folder_path):
            return []
        
        images = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                images.append(os.path.join(folder_path, file))
        return images
    
    @staticmethod
    def get_model_evaluation_images(evaluation_type):
        """Get model evaluation images - headless"""
        folder_path = MODEL_PATHS.get(evaluation_type)
        if folder_path and os.path.exists(folder_path):
            return ImageManager.get_images_from_folder(folder_path)
        return []
    
    @staticmethod
    def get_skeletal_images(skeletal_type):
        """Get skeletal structure images - headless"""
        if skeletal_type == 'human':
            return ImageManager.get_images_from_folder(HUMAN_SKELETAL_PATH)
        elif skeletal_type == 'gorilla':
            return ImageManager.get_images_from_folder(GORILLA_SKELETAL_PATH)
        return []
    
    @staticmethod
    def validate_image_paths():
        """Validate all image paths exist - headless"""
        from .config import validate_environment
        return validate_environment()