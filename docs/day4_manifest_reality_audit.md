# SatQuery AI -- Day 4 Manifest Reality Audit
## Real Physical Imagery vs Metadata-Only Verification

**Audit Standard**: Rule 1 (No Fabrication) & Rule 4 (Data Governance).
A sample is declared `REAL_LOCAL_IMAGE` ONLY if physical Sentinel-1 and Sentinel-2 GeoTIFF files exist on disk,
open without corruption, possess valid CRS and affine transforms, and contain valid remote sensing pixel values.

| Manifest | Total Records | REAL_LOCAL_IMAGE | METADATA_ONLY | MISSING_IMAGE | CORRUPT_IMAGE | Reality Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `bigearthnet_txt_manifest.json` | 4 | 4 | 0 | 0 | 0 | **READY_FOR_TRAINING** |
| `ben_train_subset.json` | 1000 | 0 | 1000 | 0 | 0 | **METADATA_ONLY_TRAINING_BLOCKED** |
| `ben_val_subset.json` | 300 | 0 | 300 | 0 | 0 | **METADATA_ONLY_TRAINING_BLOCKED** |
| `ben_test_subset.json` | 300 | 0 | 300 | 0 | 0 | **METADATA_ONLY_TRAINING_BLOCKED** |

## Audit Findings & Governance
1. **Full Benchmark Subsets (`ben_train_subset.json`, `ben_val_subset.json`, `ben_test_subset.json`)**:
   - These 1,600 samples represent deterministic, task-stratified samples from official `BigEarthNet.txt.parquet`.
   - On the current host drive `C:\`, the raw imagery (>350 GB) is **not locally extracted** to prevent disk overflow.
   - Therefore, their status is **honestly classified as `METADATA_ONLY`**.
   - They are marked `image_available: false` and `training_ready: false` in the manifests.
2. **Development Triplet Manifest (`bigearthnet_txt_manifest.json`)**:
   - Contains 4 real Sentinel-1 and Sentinel-2 paired GeoTIFF patches (`data/external/bigearthnet_txt/samples/`).
   - Audited status: **100% `REAL_LOCAL_IMAGE`** with verified CRS, dimensions, and zero corruption.
3. **External Mount Policy**:
   - To train on the 1,000-sample training subset, users mount external high-capacity storage via:
     `SATQUERY_DATA_ROOT=D:\SatQueryData` (Windows) or `export SATQUERY_DATA_ROOT=/data/satquery` (Linux).
   - No metadata-only sample is ever fed into the learned training loop.

**Active Data Root**: `C:\Users\AAKASH.S.S\OneDrive\Desktop\SatQuery AI\data`

## Audit Findings & Governance
1. **Full Benchmark Subsets (`ben_train_subset.json`, `ben_val_subset.json`, `ben_test_subset.json`)**:
   - These 1,600 samples represent deterministic, task-stratified samples from official `BigEarthNet.txt.parquet`.
   - On the current host drive `C:\`, the raw imagery (>350 GB) is **not locally extracted** to prevent disk overflow.
   - Therefore, their status is **honestly classified as `METADATA_ONLY`**.
   - They are marked `image_available: false` and `training_ready: false` in the manifests.
2. **Development Triplet Manifest (`bigearthnet_txt_manifest.json`)**:
   - Contains 4 real Sentinel-1 and Sentinel-2 paired GeoTIFF patches (`data/external/bigearthnet_txt/samples/`).
   - Audited status: **100% `REAL_LOCAL_IMAGE`** with verified CRS, dimensions, and zero corruption.
3. **External Mount Policy**:
   - To train on the 1,000-sample training subset, users mount external high-capacity storage via:
     `SATQUERY_DATA_ROOT=D:\SatQueryData` (Windows) or `export SATQUERY_DATA_ROOT=/data/satquery` (Linux).
   - No metadata-only sample is ever fed into the learned training loop.

**Active Data Root**: `C:\Users\AAKASH.S.S\OneDrive\Desktop\SatQuery AI\data`
