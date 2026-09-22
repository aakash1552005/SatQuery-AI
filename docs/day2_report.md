# SatQuery AI -- Day 2 Milestone Report
**Milestone**: `day-2-stable`  
**Date**: 2026-09-21  
**Scope**: Natural Language Query Contract, Sensor-Aware Agentic Router, SAR-vs-Optical Separation, Capability Registry, Factual Execution Trace Engine, Web GUI Query Integration  

---

## 1. Executive Summary
Day 2 delivers the core decision intelligence and sensor-aware routing engine of SatQuery AI. In strict compliance with the **Honesty Rule (Section 0)** and **SAR Separation Rule (Section 9)**, the system dynamically routes different queries and sensor modalities to transparent, dedicated pathways. SAR imagery is routed strictly to SAR deterministic radar backscatter analysis—preventing false claims of "SAR VLM" from pseudo-RGB optical conversions. Missing inputs (such as single-image temporal queries per DEMO 6) trigger precise, informative refusals without hallucinating data. All actions generate a verifiable, factual execution trace (Section 24).

---
## 2. Capabilities & Acceptance Status

| Capability / Component | Specification Section | Status | Verification / Evidence |
| :--- | :--- | :--- | :--- |
| **Natural Language Query Contract** | Sec 14 | **IMPLEMENTED & TESTED** | `src/contracts/query_contracts.py`, `QueryParser` tested across all task intents (heuristic confidence) |
| **Sensor-Aware Agentic Router** | Sec 9, Sec 15 | **IMPLEMENTED & TESTED** | `AgenticRouter` dynamically evaluates query intent and sensor modality |
| **SAR-vs-Optical Separation** | Sec 9 | **IMPLEMENTED & TESTED** | SAR inputs strictly route to `SAR_DETERMINISTIC_TOOLS` pathway, never optical VLM |
| **Input Sufficiency & Refusal Gate** | Sec 15, DEMO 6 | **IMPLEMENTED & TESTED** | 1-image temporal change correctly refused with explicit technical explanation |
| **Multimodal Fusion Routing** | Sec 15, Sec 18 | **IMPLEMENTED & TESTED** | Verified dual-modality validation (Optical + SAR) before routing to fusion engine |
| **Capability Registry State Machine** | Sec 10, Sec 23 | **IMPLEMENTED & TESTED** | Multi-dimensional readiness tracking (`routing_readiness`, `execution_readiness`, `model_availability`, `compute_readiness`) |
| **Factual Execution Trace Engine** | Sec 24 | **IMPLEMENTED & TESTED** | `TraceEngine` creates chronological step logs distinguishing `EXECUTED` from `NOT_IMPLEMENTED` scheduled tools |
| **FastAPI Routing Endpoints** | Sec 25 | **IMPLEMENTED & TESTED** | `/api/query` and `/api/capabilities` live and integrated with `AnalysisResult` contract |
| **Interactive Web UI** | Sec 25 | **IMPLEMENTED & TESTED** | Query box, SIH demo workflow chips, pathway banner, trace panel, capability modal with tool status badges |
| **Deterministic Math Contracts** | Sec 9, Sec 11 | **IMPLEMENTED & TESTED** | Spectral indices (NDVI/NDWI/MNDWI), linear SAR VV/VH ratio, dB conversions in `src/analysis/numerical_math.py` |
| **Automated Test Suite** | Sec 42 | **PASSED (19/19)** | `tests/test_day1.py` (6) + `tests/test_day2.py` (8) + `tests/test_scientific_contracts.py` (5) 100% passing |

---

## 3. Sensor-Aware Routing Matrix & Execution State Distinction

| User Query Intent | Provided Input(s) | Selected Pathway | Routing Status | Execution Status | Rationale / Rule Applied |
| :--- | :--- | :--- | :--- | :--- | :--- |
| "What land cover is visible?" | 1x Optical GeoTIFF | `OPTICAL_DETERMINISTIC` | **READY** | **NOT_IMPLEMENTED** (Day 3) | Deterministic optical analysis pathway; VLM Not Used |
| "What is visible in this scene?" | 1x SAR (VV/VH) GeoTIFF | `SAR_DETERMINISTIC_TOOLS` | **READY** | **NOT_IMPLEMENTED** (Day 3) | **Section 9**: Radar backscatter analysis; no optical hallucination |
| "Where is the largest water body?" | 1x Optical GeoTIFF | `SINGLE_IMAGE_GROUNDING` | **READY** | **NOT_IMPLEMENTED** (Day 4) | Spatial ROI extraction & bounding box localization |
| "What changed between 2024 and 2025?" | 1x Optical GeoTIFF | `REFUSAL` | **READY** | **EXECUTED** (Refusal) | **DEMO 6 / Sec 15**: Refusal; exactly 2 temporal observations required |
| "What changed between 2024 and 2025?" | 2x Temporal GeoTIFFs | `TEMPORAL_CHANGE_ENGINE` | **READY** | **NOT_IMPLEMENTED** (Day 5) | Registration gate + L1/L2 distinction |
| "Fuse optical and SAR for water" | 1x Optical + 1x SAR | `OPTICAL_SAR_FUSION` | **READY** | **NOT_IMPLEMENTED** (Day 6) | Cross-modal agreement tier analysis |
| "Fuse optical and SAR for water" | 1x Optical only | `REFUSAL` | **READY** | **EXECUTED** (Refusal) | Missing SAR modality for fusion request |

---

## 4. Test Results
```text
tests/test_day1.py::test_health_and_status PASSED                        [  5%]
tests/test_day1.py::test_raster_inspector_optical PASSED                 [ 10%]
tests/test_day1.py::test_raster_inspector_multispectral PASSED           [ 15%]
tests/test_day1.py::test_raster_inspector_sar PASSED                     [ 21%]
tests/test_day1.py::test_compatibility_temporal_pair PASSED              [ 26%]
tests/test_day1.py::test_upload_and_compatibility_api PASSED             [ 31%]
tests/test_day2.py::test_query_parser_classification PASSED              [ 36%]
tests/test_day2.py::test_capability_registry PASSED                      [ 42%]
tests/test_day2.py::test_sar_pathway_separation PASSED                   [ 47%]
tests/test_day2.py::test_missing_input_refusal_demo6 PASSED              [ 52%]
tests/test_day2.py::test_multimodal_fusion_routing PASSED                [ 57%]
tests/test_day2.py::test_execution_trace_engine PASSED                   [ 63%]
tests/test_day2.py::test_api_capabilities_endpoint PASSED                [ 68%]
tests/test_day2.py::test_api_query_flow PASSED                           [ 73%]
tests/test_scientific_contracts.py::test_spectral_indices_known_values PASSED [ 78%]
tests/test_scientific_contracts.py::test_spectral_indices_zero_denominator PASSED [ 84%]
tests/test_scientific_contracts.py::test_spectral_indices_nodata PASSED   [ 89%]
tests/test_scientific_contracts.py::test_sar_db_linear_roundtrip PASSED  [ 94%]
tests/test_scientific_contracts.py::test_sar_polarization_ratio_physics PASSED [100%]
======================== 19 passed in 0.88s ========================
```

---

## 5. Blockers & Fallback Records
- **Fallback Policy**: GeoChat VLM is bypassed via capability status checks. Optical deterministic numerical analysis is available where implemented (numerical math contracts for NDVI, NDWI, MNDWI verified in `src/analysis/numerical_math.py`; end-to-end query integration scheduled for Day 3). SAR requests currently route to the deterministic SAR pathway, but the concrete SAR analysis tools remain `NOT_IMPLEMENTED` until Day 3.
- **GeoChat Structured Preflight Record**:
  - `repository`: `ABSENT`
  - `dependencies`: `MISSING`
  - `model_weights`: `ABSENT`
  - `CUDA`: `UNAVAILABLE` (Host is Profile D CPU-only)
  - `GPU_VRAM`: `NOT_AVAILABLE` (No CUDA GPU installed on host)
  - `environment_preflight`: `NOT_EXECUTED` (Scheduled Day 3)
  - `real_model_inference`: `NOT_EXECUTED` (Never report mock/stub execution as real inference)
  - `final_capability`: `UNAVAILABLE`
- **ChangeChat**: Remains gated as `BLOCKED_LICENSE` pending licensing clearance per Section 4.1.
- **SAR Analysis Tool Implementation**: In Day 2, the pathway is routed and tool sequence scheduled (`RasterInspector`, `SARBackscatterAnalysis`, `LeeSpeckleFilter`, `PolarizationRatioEstimator`, `SARStructuredResponseComposer`). `RasterInspector` is `EXECUTED`; analysis tools are honestly tagged `NOT_IMPLEMENTED` pending Day 3 execution.
- **Synthetic Data Disclaimer**: Synthetic GeoTIFFs validate pipeline plumbing, metadata, and routing. They are engineering validation assets and do NOT support claims of real-world remote sensing accuracy.

---

## 6. Next Steps (Day 3 Scope)
- Section 4.1 & Section 8.5: GeoChat preflight test suite + SAR deterministic feature tools (`src/analysis/sar_tools.py`).
- Implement Lee speckle filtering in linear power domain ($\text{dB} \rightarrow \text{linear} \rightarrow \text{filter} \rightarrow \text{dB}$).
- Implement deterministic VV/VH backscatter statistics, linear polarization ratio ($10^{(VV_{dB} - VH_{dB})/10}$), and scene-adaptive water thresholding.
- Milestone: Full deterministic SAR VQA + Optical VQA baseline execution. Checkpoint: `day-3-stable`.
