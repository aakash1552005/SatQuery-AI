import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# Source directory for logs
conv_id = "97e89799-5e83-4090-97b0-c2409879859f"
src_log_dir = Path(f"C:/Users/AAKASH.S.S/.gemini/antigravity-ide/brain/{conv_id}/.system_generated/logs")

# Target directory in workspace
dest_dir = Path("docs/chat_history")
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Exporting session logs from: {src_log_dir}")
print(f"Saving to workspace: {dest_dir}")

# 1. Copy raw JSONL transcripts
for f in ["transcript.jsonl", "transcript_full.jsonl"]:
    src_file = src_log_dir / f
    if src_file.exists():
        dest_file = dest_dir / f
        shutil.copy2(src_file, dest_file)
        print(f"  [COPIED] {f} ({dest_file.stat().st_size / (1024**2):.2f} MB)")

# 2. Parse transcript into a human-readable Markdown conversation log
md_log_file = dest_dir / "CHAT_HISTORY_TRANSCRIPT.md"
print(f"Generating human-readable transcript: {md_log_file}")

user_count = 0
assistant_count = 0

with open(dest_dir / "transcript.jsonl", "r", encoding="utf-8") as in_f, \
     open(md_log_file, "w", encoding="utf-8") as out_f:

    out_f.write("# SatQuery AI — Complete Session Chat History & Transcript\n\n")
    out_f.write(f"- **Conversation ID**: `{conv_id}`\n")
    out_f.write(f"- **Export Timestamp**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
    out_f.write(f"- **Repository**: `SatQuery AI` (SIH Problem Statement 26167 - ISRO / SAC)\n")
    out_f.write(f"- **Team**: SatSense (ID: 148995)\n\n")
    out_f.write("---\n\n")

    for line_idx, line in enumerate(in_f):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue

        step_type = entry.get("type", "")
        source = entry.get("source", "")
        content = entry.get("content", "")
        step_idx = entry.get("step_index", line_idx)

        if step_type == "USER_INPUT" or source == "USER_EXPLICIT":
            if not content.strip():
                continue
            user_count += 1
            out_f.write(f"## 👤 User Message #{user_count} (Step {step_idx})\n\n")
            out_f.write(f"```text\n{content.strip()}\n```\n\n")
            out_f.write("---\n\n")

        elif step_type == "PLANNER_RESPONSE" or source == "MODEL":
            # Assistant final response or key plan text
            if content and content.strip():
                # Avoid logging purely raw internal prompt loops if empty
                assistant_count += 1
                out_f.write(f"## 🤖 SatQuery AI Assistant Response #{assistant_count} (Step {step_idx})\n\n")
                out_f.write(f"{content.strip()}\n\n")
                out_f.write("---\n\n")

print(f"[COMPLETE] Processed {user_count} user turns and {assistant_count} assistant responses.")
print(f"Log written to: {md_log_file}")
