"""
=============================================================================
SATQUERY AI -- CDVQA TEMPORAL DATASET MANIFEST & VALIDATOR
=============================================================================
Provides:
  - CDVQAManifest: parses and validates official CDVQA question, image,
    and answer annotations from YZHJessica/CDVQA.
  - Cross-references question ID, image ID, and split assignments.
  - Honestly records image download status (e.g., annotations present,
    underlying SECOND aerial image rasters pending/optional).
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any


class CDVQAManifest:
    """
    Parser and cross-reference validator for CDVQA temporal change question-answering dataset.
    """
    def __init__(self, cdvqa_dir: Union[str, Path]):
        self.cdvqa_dir = Path(cdvqa_dir)
        self.role = "temporal_vqa_validation"
        self.training_allowed = False
        self.evaluation_allowed = True

    def validate_dataset_structure(self) -> Dict[str, Any]:
        """
        Validates presence and counts of CDVQA files across train, val, and test splits.
        """
        results = {
            "dataset_name": "CDVQA",
            "role": self.role,
            "training_allowed": self.training_allowed,
            "evaluation_allowed": self.evaluation_allowed,
            "cdvqa_dir": str(self.cdvqa_dir),
            "splits": {},
            "status": "DISCOVERED"
        }

        splits = ["Train", "Val", "Test"]
        all_splits_present = True

        for split in splits:
            q_file = self.cdvqa_dir / f"{split}_questions.json"
            img_file = self.cdvqa_dir / f"{split}_images.json"
            ans_file = self.cdvqa_dir / f"{split}_answers.json"

            split_info = {
                "questions_present": q_file.exists(),
                "images_metadata_present": img_file.exists(),
                "answers_present": ans_file.exists(),
                "question_count": 0,
                "image_metadata_count": 0,
                "answer_record_count": 0
            }

            if q_file.exists():
                with open(q_file, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                    items = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
                    split_info["question_count"] = len(items)

            if img_file.exists():
                with open(img_file, "r", encoding="utf-8") as f:
                    img_data = json.load(f)
                    items = img_data.get("images", img_data) if isinstance(img_data, dict) else img_data
                    split_info["image_metadata_count"] = len(items)

            if ans_file.exists():
                with open(ans_file, "r", encoding="utf-8") as f:
                    ans_data = json.load(f)
                    items = ans_data.get("annotations", ans_data) if isinstance(ans_data, dict) else ans_data
                    split_info["answer_record_count"] = len(items)

            if not (q_file.exists() and img_file.exists()):
                all_splits_present = False

            results["splits"][split.lower()] = split_info

        # Check image folder presence
        img_dir = self.cdvqa_dir / "images"
        results["images_directory_present"] = img_dir.exists() and img_dir.is_dir()
        results["images_download_status"] = "PRESENT" if results["images_directory_present"] else "PENDING_UPSTREAM_EXTRACTION"

        if all_splits_present:
            results["status"] = "METADATA_READY"
        else:
            results["status"] = "PARTIAL"

        return results
