"""
Pytest configuration and session fixtures for SatQuery AI.
Ensures synthetic test rasters exist in data/samples prior to test execution.
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def ensure_test_samples():
    """Generate synthetic and dev test rasters if missing on clean CI checkouts."""
    samples_dir = PROJECT_ROOT / "data" / "samples"
    optical_sample = samples_dir / "optical" / "synthetic_optical_rgb.tif"

    if not optical_sample.exists():
        try:
            from scripts.generate_test_data import main as generate_samples
            generate_samples()
        except Exception as e:
            print(f"[Warning] Could not auto-generate test samples: {e}")

    # Ensure BigEarthNet.txt development GeoTIFF samples
    ben_samples_dir = PROJECT_ROOT / "data" / "external" / "bigearthnet_txt" / "samples"
    ben_sample = ben_samples_dir / "S1A_IW_GRDH_1SDV_20170613T165043_patch_001.tif"

    if not ben_sample.exists():
        try:
            from scripts.generate_bigearthnet_samples import generate_samples as generate_ben_samples
            generate_ben_samples()
        except Exception as e:
            print(f"[Warning] Could not auto-generate BigEarthNet samples: {e}")

    # Ensure BigEarthNet parquet metadata stub for CI
    ben_parquet = PROJECT_ROOT / "data" / "external" / "bigearthnet_txt" / "metadata" / "BigEarthNet.txt.parquet"
    if not ben_parquet.exists():
        try:
            ben_parquet.parent.mkdir(parents=True, exist_ok=True)
            import pyarrow as pa
            import pyarrow.parquet as pq
            n = 200
            records = {
                "ID": [f"ben_s1_s2_{i}" for i in range(n)],
                "s1_name": [f"S1B_IW_GRDH_1SDV_20170613T165043_33UUP_26_{i:02d}" for i in range(n)],
                "patch_id": [f"S2A_MSIL2A_20170613T101031_33UUP_26_{i:02d}" for i in range(n)],
                "input": ["What is the dominant land cover class in this agricultural parcel?" for _ in range(n)],
                "output": ["Arable land with patchy coniferous forest" for _ in range(n)],
                "latitude": [52.5 + (i * 0.001) for i in range(n)],
                "longitude": [13.3 + (i * 0.001) for i in range(n)],
                "split": ["train" for _ in range(n)],
                "type": ["binary" for _ in range(n)],
            }
            table = pa.Table.from_pydict(records)
            pq.write_table(table, ben_parquet)
        except Exception as e:
            print(f"[Warning] Could not auto-generate BigEarthNet parquet: {e}")

    # Ensure CDVQA metadata stubs for CI
    cdvqa_dir = PROJECT_ROOT / "data" / "external" / "cdvqa"
    for split in ["Train", "Val", "Test"]:
        q_file = cdvqa_dir / f"{split}_questions.json"
        img_file = cdvqa_dir / f"{split}_images.json"
        ans_file = cdvqa_dir / f"{split}_answers.json"
        try:
            cdvqa_dir.mkdir(parents=True, exist_ok=True)
            if not q_file.exists():
                import json
                with open(q_file, "w", encoding="utf-8") as f:
                    json.dump({"questions": [{"question_id": 1, "image_id": 1, "question": "Any change?"}]}, f)
            if not img_file.exists():
                import json
                with open(img_file, "w", encoding="utf-8") as f:
                    json.dump({"images": [{"image_id": 1, "file_name": "sample.tif"}]}, f)
            if not ans_file.exists():
                import json
                with open(ans_file, "w", encoding="utf-8") as f:
                    json.dump({"annotations": [{"question_id": 1, "answer": "yes"}]}, f)
        except Exception as e:
            print(f"[Warning] Could not auto-generate CDVQA stubs: {e}")

