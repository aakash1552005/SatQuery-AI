# SatQuery AI -- Model Licensing Audit and Dependency Governance
Section 4.1 / Section 31 of v4 Master Specification

## 1. Governance Principles
In accordance with the Honesty Rule (Section 0) and Licensing Safeguards (Section 4.1), no component or pathway in SatQuery AI may misrepresent its training lineage, distribution terms, or commercial suitability.

---

## 2. Foundation Model Lineage & License Audit

### A. GeoChat (Remote Sensing Vision-Language Model)
- **Primary Source**: MBZUAI / LLaVA-based RS-VLM
- **Base LLM**: Llama-2-7b-chat-hf (Subject to Meta Community License agreement)
- **Adapter Weights**: Apache-2.0
- **Status in SatQuery AI**: 
  - License note required in all user interfaces and audit documentation.
  - Runtime execution depends on available GPU profile (Profile A/B).
  - Attribution required. Never packaged as a proprietary foundation model.

### B. ChangeChat (Bi-temporal Remote Sensing VLM)
- **Primary Source**: Academic release (arXiv / GitHub)
- **Base Architecture / Licensing Gate**: 
  - License terms must be verified against academic vs non-commercial restrictions before model weights are loaded.
  - License Gate Status: `BLOCKED_LICENSE` by default until verified in writing against deployment targets.
  - System operates using deterministic change engine + L1/L2 distinction independent of ChangeChat availability.

### C. CROMA (Cross-Modal Pretrained Representation)
- **Primary Source**: Self-supervised optical-SAR representation learning
- **License**: Apache-2.0 / MIT (dependent on checkpoint release)
- **Status in SatQuery AI**:
  - Treated as optional multimodal enhancer for feature extraction.
  - Not in the critical operational path. Zero single-point-of-failure impact.

---

## 3. Dataset Licensing

### A. BigEarthNet.txt / BigEarthNet / BigEarthNet-S2
- **Citation**: arXiv:2603.29630 (2026)
- **Provider**: TU Berlin / Bifold
- **License**: CDLA-Permissive-1.0 (Community Data License Agreement -- Permissive, Version 1.0)
- **Usage**: Permitted for benchmark creation and PEFT adaptation. Attribution given.

### B. Sentinel-1 & Sentinel-2 Open Data
- **Provider**: European Space Agency (ESA) / Copernicus Programme
- **License**: Open Access under Copernicus Sentinel Data Policy (free, full, and open access).
- **Usage**: Satellite imagery testing, optical and SAR validation data.

### C. VRSBench (Public Evaluation Benchmark)
- **Citation**: Li et al., 2024 ("VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding")
- **Repository**: xiang709/VRSBench / lx709/VRSBench
- **Text Annotations License**: Creative Commons Attribution Non Commercial 4.0 (CC-BY-NC 4.0)
- **Source Imagery Provenance**: DOTA-v2 and DIOR datasets with individual non-commercial research conditions. Commercial usage is restricted by underlying aerial image sources.
- **Role**: Strictly public evaluation benchmark. Model training/fine-tuning is prohibited.

### D. CDVQA (Temporal Change VQA)
- **Citation**: Yuan et al., IEEE Transactions on Geoscience and Remote Sensing, 2022
- **Repository**: YZHJessica/CDVQA
- **License**: Apache-2.0 (Open research access for change detection visual question answering).
- **Underlying Imagery**: SECOND change detection dataset.

### E. SpaceNet 7 (Auxiliary Multi-Temporal Validation)
- **Source**: SpaceNet 7 Multi-Temporal Urban Development Challenge
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

### F. SEN12MS (Auxiliary Multimodal Dataset)
- **Source**: Technical University of Munich (MediaTUM)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

