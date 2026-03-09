import json
from pathlib import Path

rows = [json.loads(l) for l in Path("index_core.jsonl").read_text().splitlines() if l.strip()]
print("First 5 calc_dirs:", [r["calc_dir"] for r in rows[:5]])