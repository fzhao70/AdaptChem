"""
Wrapper Module - Export PyTorch models for C/Fortran integration.

This module provides tools for wrapping PyTorch models as TorchScript
and compiling C++ interfaces for use in legacy atmospheric chemistry codes
written in C or Fortran.
"""

from .wrapper import save_model, create_dynamic_library

__all__ = [
    'save_model',
    'create_dynamic_library',
]
