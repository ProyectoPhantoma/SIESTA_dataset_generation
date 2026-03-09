from pathlib import Path
import statistics

sizes = [f.stat().st_size for f in Path("data").glob("*/*/Rho.grid.nc")]
print("Count:", len(sizes))
print("Min:", min(sizes))
print("Max:", max(sizes))
print("Median:", statistics.median(sizes))