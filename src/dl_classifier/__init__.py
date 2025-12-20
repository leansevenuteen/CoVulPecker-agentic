"""
Deep Learning Classifier Module for Vulnerability Detection.

This module contains:
- CausalGAT model architecture
- Code embedding preprocessor (source code -> graph -> matrix)
- Classifier service that orchestrates the inference pipeline
"""

from .service import DLClassifierService
from .model import CausalGAT

__all__ = ["DLClassifierService", "CausalGAT"]


