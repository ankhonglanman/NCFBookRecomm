"""
Data processing and dataset modules for Book-NCF.
"""

from .preprocess import DataPreprocessor

try:
    from .dataset import NCFDataset
    __all__ = ["DataPreprocessor", "NCFDataset"]
except ImportError:
    __all__ = ["DataPreprocessor"]
