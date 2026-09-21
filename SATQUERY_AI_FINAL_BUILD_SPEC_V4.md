============================================================
SATQUERY AI — FINAL END-TO-END MASTER BUILD SPECIFICATION (v4)
============================================================

You are the senior AI/ML + geospatial + full-stack engineer
responsible for implementing:

SATQUERY AI
"An Interactive Vision-Language Assistant for Multimodal Remote
Sensing Image Analysis through Text Queries"

SIH Problem Statement: 26167
Sponsor context: ISRO / Department of Space / SAC

THIS IS THE FINAL, LOCKED, AUTHORITATIVE SPECIFICATION. It
supersedes every earlier draft (v1–v3). No further architectural
expansion after this point — remaining work is implementation,
testing, and the daily execution protocol in Section 44.

Read Section 0 and Section 44 before doing anything else.

============================================================
0. NON-NEGOTIABLE ENGINEERING PRINCIPLES
============================================================

Build SatQuery AI as an evidence-driven, agentic remote-sensing
analysis system. It must NOT be:

    User → generic LLM → answer

It must be:

    UNDERSTAND → CHECK → CHOOSE → ANALYZE → VERIFY → EXPLAIN

No specialist model is allowed to become a single point of failure.
Every mandatory PS capability must have an executable baseline
independent of optional large-model inference. Large RS-VLMs improve
the system when available; they must NEVER determine whether the
core application can start, accept data, validate data, route tasks,
perform deterministic analysis, produce evidence, show execution
traces, or generate reports.

THE GOLDEN RULE:

    The planned architecture is not evidence that a capability
    exists. A capability becomes "implemented" only after its code
    executes successfully on the current machine and passes its
    acceptance test. Never report a feature as complete merely
    because its code exists.

THE HONESTY RULE (new in v4):

    A pathway is only what it actually is. Feeding a pseudo-RGB
    conversion of a SAR image into an optical VLM and calling the
    result "SAR VQA" is a false claim. Deriving "a building was
    constructed" from a binary pixel-difference mask with no
    semantic classifier behind it is a false claim. Reporting an
    adaptation pipeline as "adaptation complete" when no training run
    has executed is a false claim. Every output must declare the
    real mechanism that produced it.

============================================================
1. PS REQUIREMENTS — MUST ALL BE IMPLEMENTED
============================================================

1. Remote-sensing image upload. 2. Input compatibility checking.
3. Optical/multispectral imagery. 4. SAR imagery. 5. Co-registered
optical + SAR pairs. 6. Bi-temporal imagery. 7. Single-image VQA
(sensor-aware — see Section 9). 8. At least one additional
single-image task: grounding OR captioning. 9. Bi-temporal change
understanding (physical + declared semantic level — Section 17).
10. Optical-SAR paired analysis (concrete fusion algorithm —
Section 18). 11. At least one remote-sensing-adapted VLM/VL
component. 12. Fine-tuning/adaptation using BigEarthNet.txt or an
approved alternative, with a verifiable trained/evaluated status
(Section 21). 13. Agentic orchestration. 14. Visual evidence.
15. Confidence (decomposed, not a single fabricated number —
Section 20). 16. Execution summary / observable trace.
17. Downloadable reports. 18. Public benchmark evaluation.
19. Clean train/validation/public-test/hidden-test separation.
20. Graceful failure for unsupported or incompatible inputs.

============================================================
2. FINAL ARCHITECTURE — SENSOR-AWARE
============================================================

                         SATQUERY AI
                              │
                              ▼
                    ┌──────────────────┐
                    │ Upload / Gateway │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Readiness Gate │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Query Contract    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Sensor Detection    │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
          ▼                  ▼                   ▼
      OPTICAL              SAR              MULTIMODAL
          │                  │                   │
          ▼                  ▼                   ▼
   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
   │   GeoChat   │   │  SAR VQA     │   │  Temporal Change  │
   │ VQA/Ground/ │   │  Pathway     │   │  OR               │
   │ Caption     │   │ (Sec. 9–10)  │   │  Optical-SAR      │
   └──────┬──────┘   └──────┬───────┘   │  Fusion Engine    │
          │                 │           │  (Sec. 17–18)     │
          │                 │           └─────────┬─────────┘
          └─────────────────┼─────────────────────┘
                             │
                             ▼
                    Evidence Store
                             │
                             ▼
              Verification Layer (numerical +
              spatial + registration + provenance)
                             │
                             ▼
                    Response Generator
                             │
                             ▼
                         GUI / Map
                             │
                             ▼
                     Reports / Exports

Optional enhancement (never a dependency):

        Optical + SAR → croma → task-specific head → additional evidence

CROMA must remain optional at all times.

============================================================
3. FEASIBILITY GATE — RUN BEFORE ANY MODEL WORK
============================================================

Do not assume GPU availability.

Create `scripts/system_check.py`. It must detect: OS, Python
version, PyTorch version, CUDA version/availability, GPU
count/name/total VRAM/free VRAM, CPU, RAM, disk space, Git, Git LFS,
Docker if available. Classify the machine:

    PROFILE A — ≥48GB VRAM class (A100/L20/A6000)
    PROFILE B — 12–24GB VRAM (high-end consumer GPU)
    PROFILE C — <12GB GPU or small-GPU dev machine
    PROFILE D — CPU only

Do NOT use a guessed fixed VRAM number as the sole decision
mechanism — empirically test model loading via the preflight scripts
in Section 4.

------------------------------------------------------------
3.1 Reference hardware profile — worked example
------------------------------------------------------------

Example machine: ASUS TUF Gaming laptop, RTX 3050 (4GB VRAM), 16GB
RAM, 512GB SSD, Ryzen 7 7000-series.

Classification: **PROFILE C.** This does not block the project —
it determines the execution strategy:

    LOCAL (this machine):
        Core app, FastAPI, frontend, GIS/rasterio processing,
        Deterministic Change Engine, Optical-SAR Fusion Engine,
        SAR deterministic-feature VQA pathway, Evidence Store,
        Verification, Reports, GUI — ALL fully feasible, zero GPU
        dependency.

    REMOTE (free/cheap cloud GPU burst — e.g. Colab/Kaggle T4,
    or a rented A100/L20-class instance for ChangeChat only):
        GeoChat inference (needs ~14GB fp16, ~4GB+ at 4-bit but
        too tight to run reliably on a 4GB card alongside image
        encoder activations), BigEarthNet.txt LoRA/PEFT training
        (gradients + optimizer state push well past 4GB even with
        QLoRA), ChangeChat (hard-requires ~48GB — never attempt
        locally on any consumer GPU).

    This machine's expected default runtime mode is HYBRID, and the
    project is still fully completable — the entire v3/v4 design
    exists specifically to make this true.

============================================================
4. MODEL PREFLIGHT + LICENSE GATES
============================================================

Create `scripts/test_geochat.py`, `test_changechat.py`,
`test_croma.py`. Each script is isolated; each test loads
environment → checkpoint → processor/tokenizer → one test
image/pair → runs one inference → parses output → returns PASS/FAIL
with a reason. If a model fails: STATUS = UNAVAILABLE. A failed model
must never break the application.

------------------------------------------------------------
4.1 License gates — mandatory, run BEFORE any integration work,
    BEFORE any hardware-budget clock starts
------------------------------------------------------------

**GeoChat License Chain Note (verified):** GeoChat is fine-tuned on
the LLaVA-1.5 architecture over `lmsys/vicuna-7b-v1.5`, which is
itself Llama 2 fine-tuned on ShareGPT data. GeoChat's own repository
label ("apache-2.0") covers its adapter/training code — it does NOT
erase the Llama 2 Community License terms carried by the merged base
weights underneath. Practical consequence for an SIH submission:

    1. Include a "Built with Llama 2" attribution notice if GeoChat
       is used.
    2. Never describe the stack as "fully Apache-2.0 / unencumbered"
       — describe it accurately as "Apache-2.0 adapter code over a
       Llama-2-licensed base," in docs/model_licenses.md.
    3. This is very unlikely to be a hard blocker for a student
       hackathon submission (Meta's restrictive terms target
       >700M-MAU commercial deployments), but it must be documented,
       not glossed over.

**ChangeChat License Gate:**

    1. Verify the repository's current LICENSE file AND any
       separate checkpoint/weight license terms independently — a
       permissive code license does not imply the same for weights.
    2. Confirm redistribution/use in an SIH submission is permitted.
    3. Record repository URL, code license, weight license terms,
       and verification date in docs/model_licenses.md.
    4. As of this specification, no clear LICENSE file could be
       confirmed for ChangeChat's repository — treat as
       BLOCKED_LICENSE by default until the team independently
       verifies otherwise. Do not proceed to inference preflight
       until this is resolved.
    5. The deterministic temporal change pipeline (Section 16)
       remains the official implementation regardless of this gate's
       outcome.

Run the same independent license verification for CROMA before
relying on its weights. Record all three (GeoChat, ChangeChat,
CROMA) in docs/model_licenses.md, regardless of outcome.

============================================================
5. ENVIRONMENT ISOLATION
============================================================

Do not force all research repositories into one Python environment.
Prefer `env-core`, `env-geochat`, `env-changechat`, `env-croma`.
Expose each model through a stable internal interface/API — the core
application communicates through capability contracts, never
repository-specific implementation details.

============================================================
6. RUNTIME MODES
============================================================

    runtime.mode = auto
    states: FULL_AI | HYBRID | DEMO_FALLBACK

At startup: system check → model preflight (incl. license gates) →
capability registry → runtime mode. The UI must always visibly show
current mode and exactly which capability is running on which
mechanism, e.g.:

    ● HYBRID MODE
      Core GIS / Change / Fusion / Evidence / Reports   READY (local)
      GeoChat                                            REMOTE
      SAR VQA (deterministic pathway)                     READY (local)
      ChangeChat                                          BLOCKED_LICENSE
      CROMA                                                DISABLED

Never silently substitute a generic model and pretend it is the
intended RS-VLM. Never silently substitute a deterministic pathway's
output and call it a model's output, or vice versa.

============================================================
7. CAPABILITY REGISTRY
============================================================

Models are capabilities, selected by the router, never hardcoded per
task. Use machine-safe, lowercase, ASCII-only IDs everywhere (see
Section 41 — no visually confusable Unicode characters in any
identifier, e.g. never a Cyrillic "О" where Latin "O" is meant).

    single_image_vqa_optical
      ├── geochat
      ├── approved_rs_vlm_fallback
      └── demo_fallback

    single_image_vqa_sar
      ├── sar_adapted_vlm            (only if one is actually verified)
      ├── sar_input_adapter_to_vlm   (labeled honestly — Section 9)
      └── sar_deterministic_features (always available — Section 8)

    single_image_grounding
      ├── geochat
      └── demo_fallback

    temporal_change
      ├── deterministic_change_engine   (always required — Section 16)
      ├── changechat                    (optional explanation layer)
      └── demo_fallback

    optical_sar_analysis
      ├── optical_sar_fusion_engine     (always required — Section 18)
      └── croma_enhanced_analyzer       (optional)

Every capability entry declares: name, task, supported modalities,
required/optional inputs, outputs, dependencies, availability,
device, status, fallback capability.

============================================================
8. FINAL MODEL REGISTRY
============================================================

8.1 GeoChat — primary optical specialist (VQA, region understanding,
grounding, captioning where supported). See Section 4.1 for its
license chain. Never replace with a generic LLaVA/CLIP/chat model
for ease of integration. If unavailable, activate the registered
fallback and never claim equivalence.

8.2 ChangeChat — OPTIONAL. Requires ~48GB-class VRAM per its own
published hardware requirements — this rules out essentially all
single consumer GPUs. **HYBRID mode (deterministic engine, no
ChangeChat) is the expected default outcome, not an edge case.**
Integration proceeds only if, in order: (1) License Gate passes,
(2) checkpoint legally usable, (3) Profile A compute available,
(4) preflight passes. Hard integration budget: 3 hours, starting
only after the license gate passes. If unsuccessful, mark
unavailable and continue — the SIH demo must be fully convincing
without it.

8.3 Deterministic Change Engine — MANDATORY, always required,
works fully independently of ChangeChat. See Section 16.

8.4 Optical-SAR Fusion Engine — MANDATORY, no CROMA dependency.
See Section 18.

8.5 SAR Deterministic Feature Tools — MANDATORY (new in v4). See
Section 8/9/10. Provides VV/VH statistics, VV/VH ratio, histogram,
texture/GLCM, spatial intensity patterns, and validated water/
non-water analysis directly from calibrated backscatter — the always
available baseline for SAR-image questions, independent of any VLM.

8.6 CROMA — OPTIONAL, stretch only. Produces representations, not
answers — never write "CROMA detected water" unless an actual
trained task head produced that result.

============================================================
9. SENSOR-AWARE ROUTING — SAR GETS ITS OWN PATHWAY
============================================================

The system must NOT treat:

    SAR → pseudo-RGB conversion → GeoChat → "SAR VQA"

as genuine SAR understanding. GeoChat is trained on optical imagery;
it has no grounded basis for interpreting radar backscatter
semantics from a color-mapped SAR image.

    QUERY
      │
      ▼
    Sensor Detection
      │
      ├── OPTICAL  → GeoChat (Section 13)
      ├── SAR      → SAR pathway (this section)
      └── MULTIMODAL → Pair Analyzer (Section 18)

SAR pathway, in priority order:

    1. If a genuinely SAR-adapted VLM is available and verified
       (not just "a VLM that accepted a SAR-derived image"):
       SAR Input Adapter → SAR-adapted VLM → VQA.
    2. Otherwise (the expected default): SAR deterministic feature
       tools (Section 8.5) → structured answer, composed by the
       response generator from real computed statistics, never from
       an LLM guessing at a radar image.

If a generic RGB-oriented VLM is used with a converted pseudo-RGB
SAR input, the execution trace and UI MUST label the pathway exactly
as:

    SAR pathway: SAR input adapter → VLM (not SAR-native)

never as "SAR-native VLM" or unqualified "SAR VQA," unless the model
was actually trained/adapted specifically for SAR.

============================================================
10. SAR INPUT ADAPTER (for the optional VLM path)
============================================================

    VV, VH, VV/VH → normalization → 3-channel pseudo-RGB → VLM

The report and execution trace must always declare this conversion
happened. This is a legitimate technique to get *some* VLM signal
from SAR data, but it is never equivalent to a model trained natively
on radar backscatter — label it as such every time it's used.

============================================================
11. GROUNDING
============================================================

Primary: GeoChat. Output: object, bbox, confidence, evidence image,
analysis_source. If GeoChat is unavailable, only two legitimate
outcomes exist: a genuinely valid text-guided grounding fallback, or
"GROUNDING: UNAVAILABLE." A generic object detector must never be
mislabeled as text-guided grounding.

============================================================
12. INPUT GATEWAY
============================================================

Support GeoTIFF, TIFF, PNG/JPEG only where permitted by the relevant
benchmark/demo. For each raster extract dimensions, CRS, transform,
bounds, resolution, band count/metadata, nodata, acquisition time,
modality, sensor metadata where available. Reject corrupted files,
unsupported formats, missing required inputs, incompatible pairs,
insufficient bands for the requested operation, missing timestamps
for temporal analysis.

============================================================
13. DATA READINESS GATE
============================================================

Build RasterInspector, CompatibilityChecker, TemporalChecker,
CoRegistrationChecker. For pairs check CRS, bounds, spatial overlap,
dimensions, resolution, affine transform, acquisition timestamp,
nodata, modality, band availability. If reprojection/resampling is
required, perform it explicitly and record source CRS, target CRS,
resampling method, output resolution. Never silently modify input
data.

============================================================
14. QUERY CONTRACT
============================================================

Convert natural language into structured intent, e.g.:

    {
      "intent": "temporal_change",
      "requires_temporal_pair": true,
      "requires_spatial_evidence": true,
      "requested_outputs": ["description", "change_region"],
      "measurement_required": false
    }

Supported task types: single_image_vqa_optical,
single_image_vqa_sar, single_image_grounding, single_image_caption,
temporal_change, optical_sar_analysis, unsupported.

============================================================
15. AGENTIC ROUTER
============================================================

Determines: task requested, inputs available, capability that
satisfies the task, availability on this machine, fallback,
required tool sequence. If a required input is missing, refuse with
a technical explanation:

    Cannot perform temporal change analysis.
    Reason: Only one acquisition was supplied.
    Required: Two temporally distinct observations.

Never hallucinate missing observations.

============================================================
16. TEMPORAL CHANGE PIPELINE — REGISTRATION-GATED, L1/L2-AWARE
============================================================

Deterministic-first, ChangeChat-optional flow:

                 BI-TEMPORAL QUERY
                        │
                        ▼
              Compatibility / Registration
                        │
                        ▼
             Registration Quality Gate
              (see 16.1 — AROSICS-based)
                        │
              PASS ─────┼───── WARN ─────┼───── REJECT
                        │                │        │
                        ▼                ▼        ▼
              Temporal Confound      continue    STOP —
              Check (16.2)           w/ reduced  report
                        │             reliability misalignment,
                        ▼             warning     no change
              Deterministic Change                claimed
              Engine
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Change Mask (L1)     Geometry / Area
              │                   │
              └─────────┬─────────┘
                         ▼
                  L1/L2 Declaration (16.3)
                         │
                  Evidence Store
                         │
              ┌──────────┴──────────┐
        ChangeChat available   ChangeChat NOT
        (license+GPU+preflight    available
         all passed)                  │
              │                       │
              ▼                       ▼
      Explanation layered on   Template/rule-based
      the verified mask/       explanation from the
      measurement               verified mask/measurement
              │                       │
              └──────────┬────────────┘
                          ▼
                   Verified Answer

------------------------------------------------------------
16.1 Registration Quality Gate
------------------------------------------------------------

Use AROSICS (real, open-source, Apache-2.0, GFZ Potsdam — Scheffler
et al. 2017, subpixel frequency-domain co-registration with built-in
outlier/cloud detection) or an equivalent verified method. Compute
residual/RMSE and overlap; classify PASS / WARN / REJECT.

    REJECT → STOP temporal analysis. Report the misalignment reason.
             Do not run change detection on misregistered imagery.
    WARN   → continue, but attach a reduced-reliability warning to
             every downstream claim.
    PASS   → proceed normally.

This prevents simple spatial misalignment from being interpreted as
physical change.

------------------------------------------------------------
16.2 Temporal Confound Check
------------------------------------------------------------

Check for and report: illumination difference, season, crop cycle,
atmospheric conditions, sensor geometry difference, SAR speckle,
resolution mismatch, cloud cover, nodata, processing-level
difference. Output a temporal reliability state: HIGH / MODERATE /
LOW. `T1 ≠ T2` does not automatically mean real-world change — this
check is what stands between a pixel difference and a defensible
claim.

------------------------------------------------------------
16.3 L1 vs L2 — mandatory declaration on every change answer
------------------------------------------------------------

L1 — PHYSICAL/IMAGE CHANGE (always available): pixel difference,
spectral difference, backscatter difference → change_detected,
change_region, change_mask, changed_area.

L2 — SEMANTIC CHANGE (only when an actual classifier supports it):
"building constructed," "vegetation removed," "water expanded,"
"road created." Never derive an L2 label from a binary mask alone.

If no semantic classifier is implemented, every change response
must explicitly say:

    Change detected.
    Level: PHYSICAL CHANGE
    Semantic interpretation: NOT AVAILABLE

============================================================
17. GEOCHAT / OPTICAL VQA IMPLEMENTATION
============================================================

Image + question → answer, confidence where available, model name,
evidence, provenance. Then grounding (bbox/region + overlay +
confidence + description). Captioning when supported.

============================================================
18. OPTICAL-SAR FUSION — CONCRETE ALGORITHM, NOT SIDE-BY-SIDE
============================================================

Must be a real fusion algorithm, not two results merely displayed
next to each other.

    OPTICAL → NDWI/MNDWI (only if the required bands exist) →
      optical water/feature mask
    SAR → VV/VH backscatter analysis (Section 8.5) →
      SAR water/feature mask
    Both masks → COMMON GRID (post-registration) → Spatial Agreement

Fusion confidence tiers, explicit and structured (not a single fake
percentage):

    BOTH AGREE     → HIGH
    OPTICAL ONLY   → MODERATE (cloud-free but SAR disagrees/missing)
    SAR ONLY       → MODERATE (cloud-obscured optical, SAR-only
                      evidence — this is exactly the "SAR sees
                      through clouds" case the PS cares about)
    NEITHER        → NOT PRESENT

Output structured result:

    {
      "class": "water",
      "optical_evidence": {...},
      "sar_evidence": {...},
      "fusion": {"agreement": "both_agree", "tier": "HIGH"},
      "regions": [...]
    }

Never use universal hard-coded SAR thresholds — thresholds must be
configurable, documented, and validated on development data.
The contribution of each modality must always be independently
identifiable in the output (never collapse to a single number with
no per-modality breakdown).

============================================================
19. EVIDENCE STORE
============================================================

    {
      "claim": "A water body is present",
      "source": "optical_sar_fusion",
      "evidence_type": "region",
      "bbox": [...],
      "confidence": {"evidence": "high", "fusion_agreement": "both_agree"},
      "inputs": ["sentinel2_01", "sentinel1_01"],
      "analysis_source": "deterministic"
    }

Evidence types: region, bbox, mask, measurement, metadata,
model_output, temporal_difference.

============================================================
20. CONFIDENCE ARCHITECTURE — DECOMPOSED, NEVER A FAKE SINGLE NUMBER
============================================================

Never present:

    Confidence = 91%

as if it were one calibrated probability. Instead:

    {
      "model_confidence": "...",
      "evidence_confidence": "HIGH",
      "measurement_quality": "COMPUTED",
      "registration_quality": "PASS",
      "fusion_agreement": "BOTH_AGREE",
      "analysis_source": "DETERMINISTIC"
    }

Treat these as system evidence states, not scientifically calibrated
probabilities, unless calibration has actually been performed and
measured.

============================================================
21. NUMERICAL GUARD
============================================================

The language model NEVER invents measurements — area, pixel count,
coordinates, distances, dates, resolutions, confidence values,
counts. The measurement engine computes the number; the response
generator receives it; the LLM may only verbalize a value it was
given, never a different one.

============================================================
22. VERIFICATION ENGINE
============================================================

EvidenceVerifier checks numerical consistency, region consistency,
mask existence, bbox validity, input provenance, temporal
consistency, output provenance, model/tool availability. If
verification fails, mark the result uncertain rather than silently
producing a confident answer.

============================================================
23. RS ADAPTATION — VERIFIABLE STATUS STATES (mandatory, high-risk)
============================================================

Do not conflate "we built an adaptation pipeline" with "we adapted a
model." Use explicit status states, and never claim a later state
than what actually executed:

    RS_ADAPTATION_PIPELINE_READY   ← manifest, subset, LoRA config exist
              ↓
    RS_ADAPTATION_TRAINED          ← a real training run executed and
                                       produced a checkpoint
              ↓
    RS_ADAPTATION_EVALUATED        ← baseline vs adapted results were
                                       actually measured and compared

Only TRAINED or EVALUATED count as completed adaptation evidence for
the PS's mandatory adaptation requirement. PIPELINE_READY alone does
not satisfy it — report it honestly as partial if that's as far as
the team got, and say so plainly in the final report rather than
implying more was done.

------------------------------------------------------------
23.1 Three-tier dataset strategy
------------------------------------------------------------

Do NOT download the entire BigEarthNet.txt corpus.

    Tier 1 — Metadata only: annotations, IDs, splits, task labels.
             Goal: a reproducible manifest.
    Tier 2 — Development subset: ~1,000–5,000 image pairs, sized to
             actual disk/bandwidth/GPU/time.
    Tier 3 — Task-filtered subset for the actual adaptation task
             (recommended first: image + question → answer). Example:
             1,000–3,000 train / 200–500 validation, adjusted to real
             resources. Record actual counts used — never fabricate.

------------------------------------------------------------
23.2 Execution strategy for constrained local hardware
------------------------------------------------------------

    LOCAL: metadata → subset generation → task filtering →
           train/validation manifest → training configuration
    REMOTE (free/cheap GPU burst): selected RS-VLM + subset →
           LoRA/PEFT → checkpoint → evaluation
    Bring back: adapter + evaluation results + configuration +
           checksums. Record base model, adapter config, dataset
           version, exact train/val counts, epochs/steps, learning
           rate, batch size, hardware, checkpoint, results. Report
           BASELINE and ADAPTED separately — never claim improvement
           unless measured. If training cannot complete, report that
           honestly and continue using the available RS-adapted
           model as-is (GeoChat's own pretraining already satisfies
           "at least one RS-adapted component" even if further
           LoRA adaptation stalls).

============================================================
24. EXECUTION TRACE
============================================================

    EXECUTION TRACE
    ✓ Input validation
    ✓ GeoTIFF metadata extracted
    ✓ Sensor detected: SAR
    ✓ Query classified: SINGLE_IMAGE_VQA_SAR
    ✓ Pathway: SAR deterministic feature tools (not VLM)
    ✓ Measurements verified
    ✓ Evidence overlay generated
    ✓ Final response generated

Also show the selected tools list. Never expose hidden
chain-of-thought — facts only.

============================================================
25. GUI
============================================================

Frontend: React/Next.js. Backend: FastAPI. ML: PyTorch, Transformers,
PEFT. GIS: Rasterio, GDAL, GeoPandas, Shapely, pyproj. Temporal:
AROSICS, OpenCV, scikit-image, NumPy. Visualization:
Leaflet/OpenLayers/MapLibre.

Screen 1 — Upload: modality, CRS, dimensions, resolution, bands,
timestamp, compatibility status.
Screen 2 — Query: natural-language box.
Screen 3 — Analysis: answer, decomposed confidence, evidence
overlay, before/after, masks/bboxes, execution trace, runtime mode
indicator, pathway declaration (which mechanism actually produced
this answer).
Screen 4 — Report: PDF, JSON, GeoJSON, masks.

============================================================
26. REPORT FORMAT
============================================================

report.pdf, result.json, execution_trace.json, evidence.geojson,
change_mask.tif. PDF sections: query, input data, sensor/modality
info, task selected, models/tools executed (with pathway honesty
labels), result, visual evidence, measurements, confidence
decomposition, limitations, execution trace.

============================================================
27. PUBLIC DATA / EVALUATION
============================================================

Strict TRAIN / VALIDATION / PUBLIC TEST / HIDDEN TEST separation.
Never use hidden ISRO/SAC evaluation data for training, fine-tuning,
threshold tuning, model selection, prompt optimization, or manual
parameter tuning. Public sources: RSVQA, VRSBench, CDVQA,
BigEarthNet.txt — use each benchmark's own prescribed splits. Do not
invent benchmark numbers.

============================================================
28. METRICS
============================================================

VQA (accuracy/exact-match), captioning (benchmark metric), grounding
(IoU/mAP), change (IoU/F1/precision/recall/change-QA), segmentation
(IoU/Dice), optical-SAR (task-specific), agent (routing accuracy,
task completion rate, invalid-input handling rate), system (latency,
memory, model availability). Never combine unrelated metrics into a
homemade score unless the official evaluation requires it.

============================================================
29. LEAKAGE CONTROL
============================================================

Prevent source/variant leakage across train/test splits. Document
split logic.

============================================================
30. CODE QUALITY / LOGGING
============================================================

Typed Python, Pydantic schemas, structured logging, config files,
env variables, unit + integration tests, reproducible scripts,
README, setup docs. Every execution logs: execution_id, timestamp,
input_ids, query, detected_task, selected_capability, selected_model,
fallback_used, parameters, duration, outputs, verification_status.

============================================================
31. FINAL PROJECT STRUCTURE
============================================================

satquery-ai/
├── app/{frontend,backend}/
├── src/
│   ├── controller/ contracts/ gateway/ router/ registry/
│   ├── models/{geochat,changechat,croma,sensor_adapter}/
│   ├── tools/
│   │   ├── vqa/ grounding/
│   │   ├── change/{deterministic,semantic}/
│   │   └── optical_sar/fusion/
│   ├── preprocessing/coregistration/
│   ├── evidence/ verification/truthfulness/ reporting/
├── scripts/
│   ├── system_check.py test_geochat.py test_changechat.py
│   ├── test_croma.py prepare_bigearthnet.py
├── configs/
├── data/
│   ├── demo/ benchmarks/
│   ├── samples/{optical,sar,temporal,optical_sar}/
│   ├── manifests/sample_manifest.json
│   └── outputs/
├── models/ tests/
├── docs/
│   ├── model_licenses.md
│   └── day1_report.md ... day8_final_report.md
├── requirements/
└── README.md

============================================================
32. 8-DAY IMPLEMENTATION PLAN (WITH DAILY CHECKPOINTS)
============================================================

Every day ends with the checkpoint ritual in Section 42.

DAY 1 — Core infrastructure + compute gate + test imagery. Build
repo, FastAPI, frontend shell, upload, GeoTIFF parser, metadata,
readiness gate, system_check.py. Acquire a small public Sentinel-1/
Sentinel-2 test corpus (single optical, single SAR, bi-temporal
pair, co-registered optical-SAR pair) into data/samples/ with a
recorded manifest — for engineering validation only, never presented
as equivalent to the hidden ISRO/SAC set. Milestone: upload →
inspect → validate, no VLM dependency. Checkpoint: `day-1-stable`.

DAY 2 — Query contract, sensor detection, capability registry,
router, execution trace, runtime mode. Milestone: different queries
route to different capabilities, including the SAR-vs-optical split.
Checkpoint: `day-2-stable`.

DAY 3 — GeoChat license note + preflight → VQA + grounding if
successful, else fallback. SAR deterministic feature tools (Section
8.5) — always build this regardless of GeoChat's outcome, since it
has zero GPU dependency. Do not spend the whole day on GeoChat
loading. Checkpoint: `day-3-stable`.

DAY 4 — BigEarthNet.txt Tier 1–3 subset, LoRA/PEFT pipeline
(local prep, remote GPU burst for the actual training run),
evaluation, honest status reporting per Section 23's state machine.
Checkpoint: `day-4-stable`.

DAY 5 — Temporal pipeline: registration → Registration Quality Gate
(AROSICS) → Temporal Confound Check → deterministic change engine →
L1 declaration → L2 only if a real classifier exists. ChangeChat
attempted only if license gate passed and Profile A compute is
available, hard-capped at 3 hours; expect HYBRID as the default
outcome. Checkpoint: `day-5-stable`.

DAY 6 — Optical-SAR: co-registration, NDWI/MNDWI optical evidence,
SAR backscatter evidence, common grid, concrete fusion algorithm
with agreement tiers (Section 18), visualization. CROMA optional,
never blocking. Checkpoint: `day-6-stable`.

DAY 7 — Evidence store, verification, numerical guard, confidence
decomposition, visual overlays, execution trace, reports (PDF/JSON/
GeoJSON), GUI polish. Checkpoint: `day-7-stable`.

DAY 8 — Run all 8 mandatory demos (Section 33) and the full
acceptance suite (Section 40). Freeze architecture. Fix only bugs,
integration problems, UI problems, reproducibility problems. No new
foundation model on Day 8. Checkpoint: `day-8-final`.

============================================================
33. MANDATORY DEMONSTRATION SCENARIOS (8)
============================================================

DEMO 1 — Optical VQA: image + land-cover question → GeoChat answer +
evidence + confidence + trace.
DEMO 2 — Grounding: "Where is the largest water body?" → bbox +
overlay + confidence.
DEMO 3 — Bi-temporal change: T1+T2 → registration gate → change mask
+ measurement + L1/L2 declaration + explanation (ChangeChat or
fallback).
DEMO 4 — Optical-SAR: "Identify surface water using both
observations" → optical evidence + SAR evidence + fusion + agreement
tier.
DEMO 5 — Agentic routing: different queries auto-select the correct
capability, optical vs SAR vs change vs fusion, without manual
selection.
DEMO 6 — Failure handling: one image only, "What changed?" → precise
refusal, no hallucinated second observation.
DEMO 7 — Compute fallback: GeoChat/ChangeChat disabled → HYBRID
indicator, unavailable capability named, active fallback named,
system continues.
DEMO 8 — SAR VQA: SAR image + question → SAR deterministic pathway
(or verified SAR-adapted VLM) → answer + evidence + pathway
declaration (never silently routed through an optical VLM).

============================================================
34. PRIORITY SYSTEM
============================================================

P0 — MUST WORK: upload, GeoTIFF validation, metadata, sensor
detection, query classification, router, optical VQA pathway, SAR
VQA pathway, grounding pathway, change detection with registration
gate + L1/L2, optical-SAR fusion, evidence, verification, trace,
GUI, reports, failure handling.

P1 — HIGH VALUE, completed where feasible: live GeoChat, RS
adaptation reaching TRAINED/EVALUATED, SAR-adapted VLM (if one is
found and verified), semantic (L2) change classifier, ChangeChat,
benchmark evaluation.

P2 — ENHANCEMENT ONLY AFTER P0/P1 STABLE: CROMA, advanced
calibration, richer semantic reasoning, additional sensors, offline
field pack.

P0 describes the required capability *workflow*; P1 describes the
strongest available *implementation* of that workflow. Never
sacrifice P0 to chase P1/P2.

============================================================
35. DO NOT BUILD
============================================================

VLM/LLM from scratch, a CROMA task head before core functionality
works, universal satellite support, every SAR polarization,
real-time satellite streaming, NavIC integration, VHF/offline
comms, mobile app, Kubernetes, distributed inference infrastructure,
elaborate cloud architecture, 10 different VLMs, a generic detector
mislabeled as grounding, fake SAR-VQA, fake semantic change, fake
accuracy, fake confidence — unless all P0/P1 requirements are
already stable.

============================================================
36. SCIENTIFIC AND ENGINEERING INTEGRITY RULES
============================================================

Never fabricate accuracy, IoU, confidence, area, coordinates,
benchmark results, training results, dataset counts, model
capabilities, or sensor specifications. If something was not
measured, say "not measured." If a model did not run, say
"unavailable." If a result is heuristic, label it "heuristic." If a
pathway used a fallback or adapter, label it exactly as such — never
a shortened or more impressive-sounding name. Do not hard-code
impressive numbers (e.g. "18.40 km²," "92.4% accuracy," "0.95 IoU")
unless they came from an actual executed run on actual input data.
Never assume a fixed band count, SAR polarization, resolution,
backscatter threshold, or calibration constant — use product
metadata and report the limitation when a required property is
missing.

============================================================
37. IDENTIFIER INTEGRITY CHECK
============================================================

Search the repository and this specification for visually
confusable Unicode characters in identifiers (e.g. Cyrillic "О"
where Latin "O" is meant). Normalize to plain ASCII everywhere:
model names, Python identifiers, JSON/YAML keys, env variables, CLI
arguments, capability IDs, filenames, API routes. Use lowercase
machine IDs (`geochat`, `changechat`, `croma`) separate from display
names ("GeoChat", "ChangeChat", "CROMA").

============================================================
38. DAILY CHECKPOINT DISCIPLINE
============================================================

At the end of every development day: run the day's acceptance
tests; confirm the app starts; record implemented/blocked/fallback
capabilities in docs/dayN_report.md; commit all verified changes;
tag `day-N-stable` (final day: `day-8-final`). Never begin the next
day on an uncommitted or unverified state. If a later day's work
breaks the system, restore the most recent stable tag rather than
repairing blindly on top of a broken state.

============================================================
39. ACCEPTANCE TEST — RUN BEFORE DECLARING COMPLETE
============================================================

 1. Can upload supported GeoTIFF? 2. Can inspect metadata? 3. Can
reject incompatible input? 4. Can classify a query by sensor and
task? 5. Can execute optical VQA? 6. Can execute SAR VQA via its own
declared pathway (never silently through GeoChat)? 7. Can perform
grounding? 8. Can process T1/T2 with a Registration Quality Gate
that actually rejects bad alignment? 9. Can generate a real change
mask with an explicit L1/L2 declaration? 10. Can explain change (via
ChangeChat or fallback)? 11. Can process optical+SAR with a concrete
fusion algorithm (not side-by-side display)? 12. Can demonstrate
independent contribution from each modality? 13. Can automatically
route the query? 14. Can show execution trace? 15. Can show
decomposed evidence/confidence (never a single fake percentage)?
16. Can prevent unsupported numerical claims? 17. Can export a
report? 18. Can operate when GeoChat is unavailable? 19. Can operate
when ChangeChat is unavailable? 20. Can operate in CPU/limited-GPU
fallback mode? 21. Can demonstrate BigEarthNet.txt adaptation without
downloading the entire corpus, and does it honestly report its
PIPELINE_READY/TRAINED/EVALUATED status? 22. Are hidden evaluation
datasets completely isolated? 23. Has every model's license (GeoChat
chain, ChangeChat, CROMA) been checked and recorded in
docs/model_licenses.md? 24. Does every completed day have a stable
Git tag and a dayN_report.md?

============================================================
40. FINAL SUCCESS CRITERIA
============================================================

A judge can: upload a valid image; see compatibility checking; ask a
question; see automatic sensor-aware task selection; receive an
answer via a truthfully-declared pathway; see visual evidence; ask a
grounding question; upload two temporal observations and see a
registration-gated change mask with an L1/L2 declaration; upload
optical+SAR and see both modalities independently contribute via a
real fusion algorithm; inspect decomposed confidence; inspect
execution trace; download a report; provide invalid data and get a
correct refusal; ask a SAR-specific question and see it handled by
the SAR pathway, not silently rerouted through an optical model; and
observe the system continue operating, honestly labeled, when an
optional large model is unavailable.

============================================================
41. FINAL GOVERNING PRINCIPLE
============================================================

DO NOT BUILD A COLLECTION OF AI MODELS. BUILD SATQUERY AI.

GeoChat, ChangeChat, CROMA, and any SAR-adapted VLM are specialists.
The SatQuery controller, sensor-aware router, evidence layer,
verification layer, registration quality gate, L1/L2 discipline,
data-readiness gate, fallback architecture, and honesty rules are
the actual system — and they are the entire engineering
contribution, independent of which specific specialist models end
up available on demo day.

============================================================
42. ANTIGRAVITY EXECUTION PROTOCOL — READ BEFORE STARTING
============================================================

This document is the master specification. **Do not execute all
eight days continuously in one unattended run.** Work in one-day
milestones, human-reviewed between each.

Start of each day's session: inspect the repository → read the
previous day's docs/dayN_report.md → verify the previous stable Git
tag → check the current capability registry state → determine what
is ACTUALLY implemented, not what was planned → execute only the
current day's scope (Section 32).

End of each day's session: run tests → start the app and verify it
runs → verify the day's acceptance criteria → update
docs/dayN_report.md → record blockers/fallbacks used → commit →
create the day's stable tag.

Interaction pattern:

    [paste this entire specification once]
            ▼
    "Execute Day 1 only."
            ▼
    [review repo, day1_report.md, day-1-stable tag]
            ▼
    "Execute Day 2 only."   ... continue through Day 8

Never issue "build everything, Days 1–8" as one instruction. After
each milestone, report status using exactly: IMPLEMENTED, TESTED,
NOT IMPLEMENTED, BLOCKED, FALLBACK USED. Never report a feature
complete merely because its code exists — the Golden Rule (Section 0)
governs every status report.

============================================================
43. FIRST ACTION
============================================================

DO NOT begin by downloading large models or datasets.

FIRST, write and run:

    python scripts/system_check.py

Then run model preflight tests where dependencies are available,
including both the GeoChat License Chain Note and the ChangeChat
License Gate (Section 4.1) before any inference preflight for either.

Then produce:

    SATQUERY FEASIBILITY REPORT

    Compute Profile:        A / B / C / D
    GPU / VRAM / CUDA / RAM / Disk:
    Platform:                (Windows → WSL2 recommended if applicable)

    GeoChat:                 READY / UNAVAILABLE
                              License: Apache-2.0 adapter over
                              Llama-2-licensed base — attribution
                              required if used
    ChangeChat:               READY / UNAVAILABLE / BLOCKED_LICENSE
    CROMA:                     READY / UNAVAILABLE

    SAR deterministic pathway: READY (always — no GPU dependency)
    Optical-SAR fusion engine: READY (always — no GPU dependency)
    Deterministic change engine: READY (always — no GPU dependency)

    BigEarthNet.txt:           metadata available / not available
    RS adaptation status:      PIPELINE_READY (initial)

    Selected Runtime:          FULL_AI / HYBRID / DEMO_FALLBACK

    P0 implementation plan:

Only after this report is generated should Day 1 implementation
proceed (Section 32, Section 42).

============================================================
END OF FINAL AUTHORITATIVE SPECIFICATION — v4
============================================================
