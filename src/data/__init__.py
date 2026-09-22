"""
SatQuery AI Data Module: Dataset loaders, manifest validators, and multimodal preprocessors.
"""
from src.data.bigearthnet_txt import (
    MultimodalSample,
    AlignmentResult,
    DatasetAuditReport,
    validate_multimodal_alignment,
    audit_dataset_duplicates_and_leakage,
    BigEarthNetTxtDataset,
)
from src.data.preprocessing import MultimodalRSPreprocessor

__all__ = [
    "MultimodalSample",
    "AlignmentResult",
    "DatasetAuditReport",
    "validate_multimodal_alignment",
    "audit_dataset_duplicates_and_leakage",
    "BigEarthNetTxtDataset",
    "MultimodalRSPreprocessor",
]
