"""
Inspect george_325.jld2 and export a plain HDF5 copy.
Replicates inspect_jld2.jl without requiring Julia.
"""

import h5py
import numpy as np
import os

jld2_path = os.path.join("Data", "datasets", "george_325.jld2")
h5_path   = os.path.join("Data", "datasets", "george_325_plain.h5")

print("=" * 60)
print(f"Loading: {jld2_path}")
print("=" * 60)

with h5py.File(jld2_path, "r") as f:
    print(f"\nKeys in file: {list(f.keys())}\n")

    arrays = {}
    for key in f.keys():
        arr = f[key][()]
        arrays[key] = arr
        nz  = np.count_nonzero(arr)
        tot = arr.size
        print(f"── {key} {'─'*(40-len(key))}")
        print(f"  shape    : {arr.shape}")
        print(f"  dtype    : {arr.dtype}")
        print(f"  nonzero  : {nz} / {tot}")
        if nz > 0:
            print(f"  min      : {arr.min():.6g}")
            print(f"  max      : {arr.max():.6g}")
            print(f"  mean     : {arr.mean():.6g}")
            # Sample a few values
            flat = arr.flatten()
            nz_vals = flat[flat != 0]
            print(f"  first nonzero values: {nz_vals[:5]}")
        else:
            print(f"  *** ALL ZEROS ***")
            # Check raw bytes too
            raw_bytes = arr.view(np.uint8)
            print(f"  nonzero bytes: {np.count_nonzero(raw_bytes)} / {raw_bytes.size}")
        print()

# ── Write plain HDF5 ──────────────────────────────────────────────────────
print("=" * 60)
print(f"Writing plain HDF5 to: {h5_path}")
print("=" * 60)

ns     = np.arange(0.05, 1.01, 0.05)
omegas = np.arange(0.5,  2.01, 0.5)
betas  = np.arange(5.0,  21.0, 1.0)

with h5py.File(h5_path, "w") as fh:
    for key, arr in arrays.items():
        ds = fh.create_dataset(key, data=arr)

    fh.attrs["ns"]     = ns
    fh.attrs["Omegas"] = omegas
    fh.attrs["betas"]  = betas
    fh["G_r"].attrs["axes"] = "bins × ntau × betas × Omegas × ns  (Julia col-major reversed)"
    fh["dos"].attrs["axes"] = "ws × betas × Omegas × ns"
    fh["ws"].attrs["axes"]  = "omega grid"

print("Done.")
