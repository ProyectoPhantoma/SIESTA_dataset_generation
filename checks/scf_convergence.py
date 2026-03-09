import json
from pathlib import Path

rows = [json.loads(l) for l in Path("index_core.jsonl").read_text().splitlines() if l.strip()]
bad = [r["calc_dir"] for r in rows if not r.get("scf_converged", True)]
print("Non-converged:", len(bad))