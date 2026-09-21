"""
Comprehensive API & Routing Verification Script for SatQuery AI
Verifies:
1. /api/health
2. /api/status
3. /api/capabilities
4. /api/upload
5. /api/compatibility
6. /api/query:
   - Optical query with optical image
   - SAR query with SAR image (verifying honest tool execution states)
   - Temporal query with 1 image -> REFUSAL (Demo 6)
   - Optical-SAR query without both modalities -> REFUSAL
   - Valid Optical-SAR query -> FUSION routing
   - Execution trace generation and format
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from app.backend.main import app

client = TestClient(app)

print("=" * 60)
print("SATQUERY AI -- INTEGRITY AUDIT VERIFICATION")
print("=" * 60)

# 1. Health check
res = client.get("/api/health")
assert res.status_code == 200, f"Health check failed: {res.text}"
print(f"[PASS] /api/health: {res.json()}")

# 2. Status check
res = client.get("/api/status")
assert res.status_code == 200, f"Status check failed: {res.text}"
status_data = res.json()
print(f"[PASS] /api/status: runtime_mode={status_data['runtime_mode']}, status={status_data['status']}")

# 3. Capabilities check
res = client.get("/api/capabilities")
assert res.status_code == 200, f"Capabilities check failed: {res.text}"
caps = res.json()["capabilities"]
assert "single_image_vqa_sar" in caps
assert caps["single_image_vqa_sar"]["routing_readiness"] == "READY"
assert "NOT_IMPLEMENTED" in caps["single_image_vqa_sar"]["execution_readiness"]
assert caps["single_image_vqa_sar"]["status"] == "NOT_IMPLEMENTED"
print(f"[PASS] /api/capabilities: verified multi-dimensional readiness (SAR routing={caps['single_image_vqa_sar']['routing_readiness']}, execution={caps['single_image_vqa_sar']['execution_readiness']})")

# 4. Upload test files
samples_dir = root / "data" / "samples"
with open(samples_dir / "optical" / "synthetic_optical_rgb.tif", "rb") as f:
    res_opt = client.post("/api/upload", files={"file": ("synthetic_optical_rgb.tif", f, "image/tiff")})
assert res_opt.status_code == 200
opt_id = res_opt.json()["file_id"]
print(f"[PASS] /api/upload (optical): file_id={opt_id}")

with open(samples_dir / "sar" / "synthetic_sar_vv_vh.tif", "rb") as f:
    res_sar = client.post("/api/upload", files={"file": ("synthetic_sar_vv_vh.tif", f, "image/tiff")})
assert res_sar.status_code == 200
sar_id = res_sar.json()["file_id"]
print(f"[PASS] /api/upload (SAR): file_id={sar_id}")

with open(samples_dir / "temporal" / "temporal_t1_2024_jan.tif", "rb") as f:
    res_t1 = client.post("/api/upload", files={"file": ("temporal_t1_2024_jan.tif", f, "image/tiff")})
assert res_t1.status_code == 200
t1_id = res_t1.json()["file_id"]

with open(samples_dir / "temporal" / "temporal_t2_2025_jan.tif", "rb") as f:
    res_t2 = client.post("/api/upload", files={"file": ("temporal_t2_2025_jan.tif", f, "image/tiff")})
assert res_t2.status_code == 200
t2_id = res_t2.json()["file_id"]

# 5. Compatibility check
res_comp = client.post("/api/compatibility", json={"file_id_a": t1_id, "file_id_b": t2_id})
assert res_comp.status_code == 200
assert res_comp.json()["crs_match"] is True
print(f"[PASS] /api/compatibility: {res_comp.json()['status']}")

# 6. Test Day 2 Key Behaviors:

# D1. Optical query with optical image
res_q1 = client.post("/api/query", json={
    "query": "What type of land cover dominates this optical scene?",
    "file_ids": [opt_id]
})
assert res_q1.status_code == 200
data_q1 = res_q1.json()
assert data_q1["decision"]["task_type"] == "single_image_vqa_optical"
assert data_q1["decision"]["pathway"] == "optical_deterministic"
assert data_q1["result"]["mechanism"] == "deterministic_optical_spectral_analysis"
assert data_q1["result"]["model"] is None
print(f"[PASS] D1: Optical Query -> Routed to Optical Deterministic, mechanism={data_q1['result']['mechanism']}, VLM=None")

# D2. SAR query with SAR image
res_q2 = client.post("/api/query", json={
    "query": "Analyze radar backscatter and surface roughness in this SAR image.",
    "file_ids": [sar_id]
})
assert res_q2.status_code == 200
data_q2 = res_q2.json()
assert data_q2["decision"]["task_type"] == "single_image_vqa_sar"
assert data_q2["decision"]["pathway"] == "sar_deterministic_tools"
# Verify honest tool execution status
tools_exec = {t["tool_name"]: t["status"] for t in data_q2["decision"]["tool_executions"]}
assert tools_exec["RasterInspector"] == "EXECUTED"
assert any("SARBackscatterAnalysis" in k and v == "NOT_IMPLEMENTED" for k, v in tools_exec.items())
assert tools_exec["LeeSpeckleFilter"] == "NOT_IMPLEMENTED"
assert data_q2["result"]["mechanism"] == "deterministic_sar_analysis"
print(f"[PASS] D2: SAR Query -> Routed to SAR Deterministic Tools, tools_executed={tools_exec}")

# D3. Temporal query with only one image -> REFUSAL
res_q3 = client.post("/api/query", json={
    "query": "What changed in this area between the two dates?",
    "file_ids": [opt_id]
})
assert res_q3.status_code == 200
data_q3 = res_q3.json()
assert data_q3["decision"]["is_executable"] is False
assert data_q3["decision"]["pathway"] == "refusal"
assert "Only one acquisition was supplied" in data_q3["decision"]["refusal_reason"]
print(f"[PASS] D3: Single-image temporal query -> Refused honestly: '{data_q3['decision']['refusal_reason']}'")

# D4. Optical-SAR query without both modalities -> REFUSAL
res_q4 = client.post("/api/query", json={
    "query": "Fuse optical and SAR observations to map water bodies.",
    "file_ids": [opt_id]
})
assert res_q4.status_code == 200
data_q4 = res_q4.json()
assert data_q4["decision"]["is_executable"] is False
assert "Cannot perform Optical-SAR cross-modal fusion" in data_q4["decision"]["refusal_reason"]
print(f"[PASS] D4: Optical-SAR query without SAR -> Refused honestly: '{data_q4['decision']['refusal_reason']}'")

# D5. Valid optical-SAR query -> ROUTING
res_q5 = client.post("/api/query", json={
    "query": "Fuse optical and SAR observations to map water bodies.",
    "file_ids": [opt_id, sar_id]
})
assert res_q5.status_code == 200
data_q5 = res_q5.json()
assert data_q5["decision"]["is_executable"] is True
assert data_q5["decision"]["pathway"] == "optical_sar_fusion"
print(f"[PASS] D5: Valid Optical + SAR query -> Routed to optical_sar_fusion pathway")

# D6. Execution trace generation
trace = data_q2["trace"]
assert len(trace["trace_id"]) > 0
step_names = [s["name"] for s in trace["steps"]]
assert "Input validation" in step_names
assert "Query classified" in step_names
assert "Pathway selected" in step_names
assert "Tool execution sequence prepared" in step_names
print(f"[PASS] D6: Execution trace generated successfully with trace_id='{trace['trace_id']}' and steps: {step_names}")

print("=" * 60)
print("ALL DAY 1 & DAY 2 INTEGRITY VERIFICATION CHECKS PASSED!")
print("=" * 60)
