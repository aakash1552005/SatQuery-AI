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
    """Generate synthetic test rasters if missing on clean CI checkouts."""
    samples_dir = PROJECT_ROOT / "data" / "samples"
    optical_sample = samples_dir / "optical" / "synthetic_optical_rgb.tif"

    if not optical_sample.exists():
        try:
            from scripts.generate_test_data import main as generate_samples
            generate_samples()
        except Exception as e:
            print(f"[Warning] Could not auto-generate test samples: {e}")
