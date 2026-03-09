import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("index_core.jsonl").read_text().splitlines() if l.strip()]
c = Counter(r["element"] for r in rows)
print(c)