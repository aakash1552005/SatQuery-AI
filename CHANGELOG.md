# Changelog

All notable changes to **SatQuery AI (SatSense)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.0.0] - 2026-10-06 — Production Architecture Freeze (SIH 2026 PS 26167)

### Added
- **Global Edge Terminal**: Deployed live mission control interface to Cloudflare Workers Static Assets at [`https://satquery-ai.aakash1552005.workers.dev/`](https://satquery-ai.aakash1552005.workers.dev/).
- **Aerospace GIS Workstation UI**:
  - Rebuilt `app/frontend/index.html` adhering to aerospace defense terminal standards.
  - Custom geometric satellite aperture vector favicon at `/favicon.svg`.
  - Dynamic Telemetry Switcher supporting local backends (`http://127.0.0.1:8000`), secure Cloudflare tunnels, and remote GPU endpoints.
  - Pure SVG telemetry indicators, tactical export trays, and live agreement matrix visualization.
- **Deterministic Scientific GIS Core**:
  - `src/analysis/sar_tools.py`: Linear power domain Refined Lee Speckle Filtering ($10^{\text{dB}/10}$), bounded Otsu thresholding $[-25.0, -10.0\text{ dB}]$, and dual-polarization cross-ratio ($\sigma^0_{VV} - \sigma^0_{VH}$).
  - `src/analysis/optical_tools.py`: Zero-division guarded spectral indices (NDVI, NDWI, MNDWI) and 5-class rule-based spectral land cover classifier.
  - `src/analysis/fusion_engine.py`: Optical-SAR cross-modal consensus and physical agreement matrix generation.
  - `src/analysis/change_engine.py`: Bi-temporal physical difference engine with L1 radiometric change gating.
- **Careful Coordinator & Verification Subsystem**:
  - `src/router/router.py`: Sensor-aware query parsing, strict SAR-optical separation, and sufficiency refusal gates.
  - `src/verification/numerical_guard.py`: Deterministic token auditor locking LLM text strictly to certified GIS calculations (0.0% area error).
  - `src/reporting/field_pack.py`: 1-Click Air-Gapped Field Pack exporter bundling GeoJSON (RFC 7946), offline HTML reports, and provenance checksums.
- **Comprehensive SIH 2026 Presentation Package**:
  - `sih_presentation/SatQuery_AI_SIH2026_SatSense_V2.pptx`: Official 16:9 widescreen presentation deck.
  - `sih_presentation/index.html`: Interactive web presentation with timer and speaker notes drawer.
  - `sih_presentation/SPEAKER_NOTES_AND_DEFENSE_GUIDE.md`: Word-for-word 3-minute pitch script and 10 lethal jury question defenses.
- **Automated Test Suite**:
  - 14 test suites covering 118 unit and integration tests passing at 100% in 19.2s (`pytest tests/`).
- **Open Source Standards**:
  - Apache-2.0 License (`LICENSE`).
  - Standard PEP 517/621 packaging (`pyproject.toml`).
  - GitHub Actions automated CI workflow (`.github/workflows/ci.yml`).
