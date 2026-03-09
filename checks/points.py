import json
from pathlib import Path

root = Path("data")
rows = [json.loads(l) for l in Path("index_core.jsonl").read_text().splitlines() if l.strip()]

missing = []
for r in rows:
    cd = root / r["calc_dir"]
    for f in ["Rho.grid.nc", "ElectrostaticPotential.grid.nc"]:
        if not (cd / f).exists():
            missing.append((r["calc_dir"], f))

print("Broken index entries:", len(missing))
print(missing[:10])