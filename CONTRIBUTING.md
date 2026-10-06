# Contributing to SatQuery AI

Thank you for your interest in contributing to **SatQuery AI (SatSense)**!

SatQuery AI is engineered for mission-critical Earth Observation (EO) under Smart India Hackathon 2026 for the Space Applications Centre (SAC), Indian Space Research Organisation (ISRO).

---

## 1. Development Principles

All contributions must strictly follow our core engineering covenants:
1. **The Honesty Rule**: Never claim synthetic data is real spaceborne imagery. Never mock model weights or fabricate benchmark numbers.
2. **Decouple Perception from Math**: Neural networks perceive semantics; deterministic GIS affine matrices compute physical areas in $\text{km}^2$.
3. **Linear SAR Physics**: Microwave SAR decibels must always be converted to linear power ($10^{\text{dB}/10}$) before averaging, speckle filtering, or calculating ratios.
4. **Sufficiency Refusals**: Incomplete queries or missing bands must trigger an honest refusal rather than a hallucinated prediction.
5. **Numerical Token Guard**: Numerical area figures in generated language must match certified GIS execution dictionaries.

---

## 2. Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/aakash1552005/SatQuery-AI.git
cd "SatQuery AI"

# Create a virtual environment with Python 3.11
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

---

## 3. Running the Test Suite

SatQuery AI maintains 100% passing automated tests. All PRs must pass all 14 test suites:

```bash
pytest tests/ -v
```

---

## 4. Submitting a Pull Request

1. Create a feature branch: `git checkout -b feat/your-feature-name`.
2. Follow Conventional Commits:
   - `feat(analysis): add NISAR L-band calibration constant`
   - `fix(router): refine cloud-penetration refusal gate`
   - `docs(presentation): update jury defense benchmarks`
3. Ensure all tests pass.
4. Submit your PR against the `main` branch with a clear description and reproducibility steps.
