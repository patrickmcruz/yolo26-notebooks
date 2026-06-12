"""
Helper module for loading YOLO26 models from the models/ directory.

This module provides utilities for notebooks to find and load pre-trained
YOLO26 models regardless of where the notebook is executed from.

Usage:
    from models_helper import get_model_path, load_model
    
    # Get absolute path to a model
    model_path = get_model_path('yolo26m.pt')
    
    # Or load directly
    from ultralytics import YOLO
    model = YOLO(str(get_model_path('yolo26m.pt')))
"""

from pathlib import Path
from typing import Optional


def get_notebooks_root() -> Path:
    """
    Find the notebooks root directory.
    
    Returns:
        Path: Absolute path to the notebooks/ directory
    """
    # This file is in notebooks/models_helper.py
    # So its parent is notebooks/
    return Path(__file__).parent.resolve()


def get_models_dir() -> Path:
    """
    Get the models directory path.
    
    Returns:
        Path: Absolute path to notebooks/models/
    """
    notebooks_root = get_notebooks_root()
    models_dir = notebooks_root / "models"
    
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory not found: {models_dir}\n"
            f"Expected: {notebooks_root}/models/\n"
            f"Available models should be: yolo26n.pt, yolo26m.pt, yolo26x.pt"
        )
    
    return models_dir


def get_model_path(model_name: str) -> Path:
    """
    Get the absolute path to a YOLO26 model.
    
    Args:
        model_name: Name of the model file (e.g., 'yolo26m.pt')
    
    Returns:
        Path: Absolute path to the model file
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
    
    Example:
        >>> from models_helper import get_model_path
        >>> path = get_model_path('yolo26m.pt')
        >>> print(path)
        /path/to/notebooks/models/yolo26m.pt
    """
    models_dir = get_models_dir()
    model_path = models_dir / model_name
    
    if not model_path.exists():
        available = [f.name for f in models_dir.glob("*.pt")]
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Available models: {available}"
        )
    
    return model_path


def load_model(model_name: str, **kwargs):
    """
    Load a YOLO26 model using Ultralytics.
    
    Args:
        model_name: Name of the model file (e.g., 'yolo26m.pt')
        **kwargs: Additional arguments passed to YOLO constructor
    
    Returns:
        YOLO: Loaded model instance
    
    Example:
        >>> from models_helper import load_model
        >>> model = load_model('yolo26m.pt')
        >>> results = model('image.jpg')
    """
    from ultralytics import YOLO
    
    model_path = get_model_path(model_name)
    return YOLO(str(model_path), **kwargs)


def list_available_models() -> list:
    """
    List all available YOLO26 models.
    
    Returns:
        list: List of available model filenames
    
    Example:
        >>> from models_helper import list_available_models
        >>> models = list_available_models()
        >>> print(models)
        ['yolo26m.pt', 'yolo26n.pt', 'yolo26x.pt']
    """
    models_dir = get_models_dir()
    return sorted([f.name for f in models_dir.glob("*.pt")])


if __name__ == "__main__":
    # Self-test
    print("=" * 70)
    print("YOLO26 Models Helper - Self Test")
    print("=" * 70)
    
    try:
        print(f"\nNotebooks root: {get_notebooks_root()}")
        print(f"Models directory: {get_models_dir()}")
        
        available = list_available_models()
        print(f"\nAvailable models ({len(available)}):")
        for model in available:
            path = get_model_path(model)
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  - {model:20} ({size_mb:6.1f} MB) - {path}")
        
        print("\n[SUCCESS] All models accessible!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
