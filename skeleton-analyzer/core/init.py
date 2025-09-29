"""
Skeleton Analyzer Pro - Core Module
Headless computation, model logic, and data processing
"""

from .config import (
    BASE_PATH, MODEL_PATH, DATABASE_PATH,
    HUMAN_SKELETAL_PATH, GORILLA_SKELETAL_PATH,
    OUR_DATABASE_PATH, ON_DATABASE_PATH,
    MODEL_PATHS, validate_environment
)

from .model import SkeletonModel
from .database import DatabaseManager
from .preprocessing import ImageManager
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__author__ = "Skeleton Analyzer Pro Team"

__all__ = [
    # Config
    'BASE_PATH', 'MODEL_PATH', 'DATABASE_PATH',
    'HUMAN_SKELETAL_PATH', 'GORILLA_SKELETAL_PATH',
    'OUR_DATABASE_PATH', 'ON_DATABASE_PATH',
    'MODEL_PATHS', 'validate_environment',
    
    # Core classes
    'SkeletonModel',
    'DatabaseManager', 
    'ImageManager',
    'ModelEvaluator',
]