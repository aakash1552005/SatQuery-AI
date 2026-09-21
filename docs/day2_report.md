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
| **Natural Language Query Contract** | Sec 14 | **IMPLEMENTED & TESTED** | `src/contracts/query_contracts.py`, `QueryParser` tested across all task intents |
| **Sensor-Aware Agentic Router** | Sec 9, Sec 15 | **IMPLEMENTED & TESTED** | `AgenticRouter` dynamically evaluates query intent and sensor modality |
| **SAR-vs-Optical Separation** | Sec 9 | **IMPLEMENTED & TESTED** | SAR inputs strictly route to `SAR_DETERMINISTIC_TOOLS` pathway, never optical VLM |
| **Input Sufficiency & Refusal Gate** | Sec 15, DEMO 6 | **IMPLEMENTED & TESTED** | 1-image temporal change correctly refused with explicit technical explanation |
| **Multimodal Fusion Routing** | Sec 15, Sec 18 | **IMPLEMENTED & TESTED** | Verified dual-modality validation (Optical + SAR) before routing to fusion engine |
| **Capability Registry State Machine** | Sec 10, Sec 23 | **IMPLEMENTED & TESTED** | Dynamic status tracking (`READY`, `UNAVAILABLE`, `BLOCKED_LICENSE`, `FALLBACK_ACTIVE`) |
| **Factual Execution Trace Engine** | Sec 24 | **IMPLEMENTED & TESTED** | `TraceEngine` creates chronological step logs with tool sequence (no hidden CoT) |
| **FastAPI Routing Endpoints** | Sec 25 | **IMPLEMENTED & TESTED** | `/api/query` and `/api/capabilities` live and integrated |
| **Interactive Web UI** | Sec 25 | **IMPLEMENTED & TESTED** | Query box, demo preset chips, pathway banner, trace panel, capability modal |
| **Automated Test Suite** | Sec 42 | **PASSED (14/14)** | `tests/test_day1.py` + `tests/test_day2.py` 100% passing in pytest |

---

## 3. Sensor-Aware Routing Matrix

| User Query Intent | Provided Input(s) | Selected Pathway | Executable? | Rationale / Rule Applied |
| :--- | :--- | :--- | :--- | :--- |
| "What land cover is visible?" | 1x Optical GeoTIFF | `OPTICAL_DETERMINISTIC` (or `OPTICAL_VLM`) | **YES** | Standard single-image optical VQA |
| "What is visible in this scene?" | 1x SAR (VV/VH) GeoTIFF | `SAR_DETERMINISTIC_TOOLS` | **YES** | **Section 9**: Radar backscatter analysis; no optical hallucination |
| "Where is the largest water body?" | 1x Optical GeoTIFF | `SINGLE_IMAGE_GROUNDING` | **YES** | Spatial ROI extraction & bounding box localization |
| "What changed between 2024 and 2025?" | 1x Optical GeoTIFF | `REFUSAL` | **NO** | **DEMO 6 / Sec 15**: Refusal; exactly 2 temporal observations required |
| "What changed between 2024 and 2025?" | 2x Temporal GeoTIFFs | `TEMPORAL_CHANGE_ENGINE` | **YES** | Registration gate + L1/L2 distinction |
| "Fuse optical and SAR for water" | 1x Optical + 1x SAR | `OPTICAL_SAR_FUSION` | **YES** | Cross-modal agreement tier analysis |
| "Fuse optical and SAR for water" | 1x Optical only | `REFUSAL` | **NO** | Missing SAR modality for fusion request |

---

## 4. Test Results
```text
tests/test_day1.py::test_health_and_status PASSED                        [  7%]
tests/test_day1.py::test_raster_inspector_optical PASSED                 [ 14%]
tests/test_day1.py::test_raster_inspector_multispectral PASSED           [ 21%]
tests/test_day1.py::test_raster_inspector_sar PASSED                     [ 28%]
tests/test_day1.py::test_compatibility_temporal_pair PASSED              [ 35%]
tests/test_day1.py::test_upload_and_compatibility_api PASSED             [ 42%]
tests/test_day2.py::test_query_parser_classification PASSED              [ 50%]
tests/test_day2.py::test_capability_registry PASSED                      [ 57%]
tests/test_day2.py::test_sar_pathway_separation PASSED                   [ 64%]
tests/test_day2.py::test_missing_input_refusal_demo6 PASSED              [ 71%]
tests/test_day2.py::test_multimodal_fusion_routing PASSED                [ 78%]
tests/test_day2.py::test_execution_trace_engine PASSED                   [ 85%]
tests/test_day2.py::test_api_capabilities_endpoint PASSED                [ 92%]
tests/test_day2.py::test_api_query_flow PASSED                           [100%]
======================== 14 passed, 1 warning in 1.05s ========================
```

---

## 5. Blockers & Fallback Records
- **VLM Local GPU**: Profile D host has no CUDA GPU; GeoChat is marked `UNAVAILABLE` and optical VQA runs through `OPTICAL_DETERMINISTIC` baseline without disrupting application execution or routing.
- **ChangeChat**: Remains gated as `BLOCKED_LICENSE` pending licensing clearance per Section 4.1.

---

## 6. Next Steps (Day 3 Scope)
- Section 4.1 & Section 8.5: GeoChat preflight test suite + SAR deterministic feature tools (`src/analysis/sar_tools.py`).
- Implement Lee speckle filtering, VV/VH backscatter statistics, polarization ratio, and water thresholding for SAR.
- Milestone: Full deterministic SAR VQA + Optical VQA baseline execution. Checkpoint: `day-3-stable`.
