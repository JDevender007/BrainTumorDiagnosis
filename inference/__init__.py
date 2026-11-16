"""
BrainTumorDiagnosis Inference Package

High-level facade for running the complete brain tumor analysis
pipeline on single MRI cases.

Usage:

    python -m inference.predict <case_directory>
"""

from .pipeline import (
    InferencePipeline,
    InferencePipelineConfig,
    run_case_pipeline,
)

from .predict import main as predict_main


__all__ = [
    "InferencePipeline",
    "InferencePipelineConfig",
    "run_case_pipeline",
    "predict_main",
]