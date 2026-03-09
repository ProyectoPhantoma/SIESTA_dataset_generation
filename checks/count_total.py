import json
from pathlib import Path

p = Path("index_core.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
print("Total rows:", len(rows))