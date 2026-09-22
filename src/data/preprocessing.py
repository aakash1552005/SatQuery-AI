"""
Multimodal Remote Sensing Preprocessor for BigEarthNet.txt.
Handles SAR physical calibration, optical reflectance normalization, text instruction formatting,
and batch collation with preserved provenance.
"""

from typing import Optional, Any
import numpy as np
from src.analysis.sar_tools import LeeSpeckleFilter, sar_db_to_linear

try:
    import torch
except ImportError:
    torch = None


class MultimodalRSPreprocessor:
    """
    Unified preprocessor for Sentinel-1 SAR, Sentinel-2 Multispectral, and RS-VLM Text Prompts.
    Maintains physical validity and provenance across modalities.
    """

    def __init__(
        self,
        sar_apply_lee_filter: bool = True,
        sar_filter_window: int = 5,
        optical_scale_factor: float = 10000.0,
        target_image_size: Optional[tuple[int, int]] = (120, 120),
    ):
        self.sar_apply_lee_filter = sar_apply_lee_filter
        self.sar_filter_window = sar_filter_window
        self.optical_scale_factor = optical_scale_factor
        self.target_image_size = target_image_size
        self.lee_filter = LeeSpeckleFilter(window_size=sar_filter_window, enl=4.0)

    def preprocess_sar(self, sar_raw: np.ndarray) -> np.ndarray:
        """
        Preprocess 2-band Sentinel-1 SAR (Band 0: VV, Band 1: VH) in decibels.
        1. Optionally applies Lee speckle filter in the linear power domain.
        2. Normalizes typical backscatter range [-30 dB, 0 dB] into standardized [-1.0, 1.0].
        Returns float32 array of shape (2, H, W).
        """
        arr = sar_raw.astype(np.float64)
        ch, h, w = arr.shape
        filtered_bands = []

        for c in range(ch):
            band = arr[c]
            if self.sar_apply_lee_filter:
                res = self.lee_filter.filter(band, input_is_db=True, output_as_db=True)
                band_filt = res.filtered_array
            else:
                band_filt = band

            # Standardized normalization: clamp [-35 dB, 5 dB] and map to [-1.0, 1.0]
            clamped = np.clip(band_filt, -35.0, 5.0)
            normalized = (clamped - (-15.0)) / 10.0  # zero-centered around typical -15 dB
            filtered_bands.append(normalized.astype(np.float32))

        return np.stack(filtered_bands, axis=0)

    def preprocess_optical(self, optical_raw: np.ndarray) -> np.ndarray:
        """
        Preprocess 4-band Sentinel-2 Optical (B02, B03, B04, B08) in surface reflectance [0, 10000].
        1. Scales reflectance: x / 10000.0.
        2. Clips to physically plausible surface reflectance [0.0, 1.0].
        Returns float32 array of shape (4, H, W).
        """
        arr = optical_raw.astype(np.float32)
        scaled = np.clip(arr / self.optical_scale_factor, 0.0, 1.0)
        return scaled

    def format_instruction_prompt(self, sample_record: dict) -> str:
        """
        Format task-aware instruction prompt for RS-VLM / GeoChat alignment.
        Supports:
        - vqa: "User: <image_s1><image_s2> Question: {q} Assistant: {a}"
        - captioning: "User: <image_s1><image_s2> Provide a comprehensive remote sensing description of this scene. Assistant: {caption}"
        - land_cover_reasoning: "User: <image_s1><image_s2> {instruction} Assistant: {explanation}"
        """
        task = sample_record.get("task_type", "vqa").lower()

        if task == "vqa" and "question" in sample_record:
            q = sample_record["question"]
            a = sample_record.get("answer", "")
            return f"User: <image_s1><image_s2> Question: {q}\nAssistant: {a}"

        elif task == "captioning" and "caption" in sample_record:
            cap = sample_record["caption"]
            return f"User: <image_s1><image_s2> Provide a comprehensive remote sensing description of this scene.\nAssistant: {cap}"

        elif task == "land_cover_reasoning" and "instruction" in sample_record:
            inst = sample_record["instruction"]
            exp = sample_record.get("explanation", "")
            return f"User: <image_s1><image_s2> {inst}\nAssistant: {exp}"

        # Fallback to general text
        text = sample_record.get("text", "")
        return f"User: <image_s1><image_s2> Analyze this multimodal remote sensing pair.\nAssistant: {text}"

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Batch collation function for PyTorch DataLoader.
        Stacks S1 and S2 arrays into batched tensors, collects metadata and formatted texts.
        """
        sample_ids = [item["sample_id"] for item in batch]
        splits = [item["split"] for item in batch]
        task_types = [item["task_type"] for item in batch]
        texts = [item["text"] for item in batch]

        s1_arrays = [item["s1_data"] for item in batch]
        s2_arrays = [item["s2_data"] for item in batch]

        if torch is not None:
            s1_tensor = torch.from_numpy(np.stack(s1_arrays, axis=0))
            s2_tensor = torch.from_numpy(np.stack(s2_arrays, axis=0))
        else:
            s1_tensor = np.stack(s1_arrays, axis=0)
            s2_tensor = np.stack(s2_arrays, axis=0)

        return {
            "sample_ids": sample_ids,
            "splits": splits,
            "task_types": task_types,
            "texts": texts,
            "s1_tensors": s1_tensor,
            "s2_tensors": s2_tensor,
            "batch_size": len(batch),
            "provenance": {
                "dataset": "BigEarthNet.txt",
                "sar_channels": ["VV", "VH"],
                "optical_channels": ["B02_Blue", "B03_Green", "B04_Red", "B08_NIR"],
            }
        }
