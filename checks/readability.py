from pathlib import Path
import random
from netCDF4 import Dataset

root = Path("data")
files = list(root.glob("*/*/Rho.grid.nc"))
sample = random.sample(files, 5)

for f in sample:
    ds = Dataset(f)
    print(f, "OK | variables:", list(ds.variables.keys())[:3])
    ds.close()