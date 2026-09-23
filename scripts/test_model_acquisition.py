import os
import sys
from pathlib import Path

print("=" * 60)
print("SATQUERY AI -- MODEL ACQUISITION & LOAD TEST")
print("=" * 60)

model_id = "MBZUAI/geochat-7b"
print(f"Target Model ID: {model_id}")

hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
print(f"HuggingFace Hub Directory: {hf_home}")

# Check if model folder exists in cache
found_cached = False
if os.path.exists(hf_home):
    for entry in os.listdir(hf_home):
        if "geochat" in entry.lower():
            print(f"Found candidate cache folder: {entry}")
            found_cached = True

if not found_cached:
    print(f"GeoChat 7B weights NOT found in local HuggingFace cache: {hf_home}")

# Check local models dir
local_model_dir = Path("models")
for p in local_model_dir.glob("**/*"):
    if p.is_file() and p.suffix in [".bin", ".safetensors", ".pt"]:
        print(f"Found local weight file: {p}")

print("\nAttempting model load test...")
try:
    from transformers import AutoConfig, AutoTokenizer
    print("Loading config...")
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    print("Config LOAD: PASS")
except Exception as e:
    print(f"Config LOAD: FAIL ({e})")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    print("Tokenizer LOAD: PASS")
except Exception as e:
    print(f"Tokenizer LOAD: FAIL ({e})")

try:
    from transformers import AutoModelForCausalLM
    print("Loading full model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    print("Model LOAD: PASS")
except Exception as e:
    print(f"Model LOAD: FAIL ({e})")

print("=" * 60)
