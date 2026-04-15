# Setup Guide: MaxEnt Benchmark Tutorial

This guide walks through everything you need to do **before** opening
`benchmark_tutorial.ipynb` for the first time.

---

## Step 1 — Download the required folder

You only need the **`MaxEnt_benchmark/`** folder from the repository.
You do not need to clone or download the entire repo.

### Option A — Clone the full repo (simplest)

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>/MaxEnt_benchmark
```

### Option B — Download just the folder (no git)

1. Go to the repository on GitHub.
2. Click **Code → Download ZIP**.
3. Unzip it and navigate into `MaxEnt_benchmark/`.

After either option, your working directory should look like this:

```
MaxEnt_benchmark/
├── benchmark_tutorial.ipynb   ← the tutorial notebook
├── setup_guide.md             ← this file
├── ana_cont/                  ← MaxEnt library source (must be installed)
│   ├── ana_cont/
│   │   └── continuation.py
│   └── setup.py
├── run_anacont_maxent.py      ← script to run MaxEnt on your own data
├── compare_vae_maxent.py      ← script to generate comparison figures
└── out/                       ← optional: pre-computed MaxEnt results
    └── anacont_<tag>/
        └── summary_mean_fullcov.npz
```

> The `out/` folder and everything inside it is optional. If it is absent,
> the notebook will skip the pre-computed MaxEnt step and ask you to run
> MaxEnt yourself instead.

---

## Step 2 — Set up a Python environment (Optional)

It is recommended to work inside a **virtual environment** so the
`ana_cont` installation does not conflict with your system Python.

### Option A — `venv` (built-in, no extra tools)

```bash
# Create the environment (once)
python3 -m venv .venv

# Activate it — run this every time you open a new terminal
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

### Option B — `conda`

```bash
# Create the environment (once)
conda create -n maxent_bench python=3.10

# Activate it
conda activate maxent_bench
```

---

## Step 3 — Install dependencies

With your environment activated, run the following from inside
`MaxEnt_benchmark/`:

```bash
# Core scientific stack — always required
pip install numpy scipy matplotlib

# Jupyter — required to open the notebook
pip install jupyterlab          # or: pip install notebook

# PyTorch 
pip install torch

# Cython — required before installing ana_cont
pip install cython

# ana_cont — the MaxEnt library bundled in this folder
# The -e flag installs it in "editable" mode so no files are copied elsewhere.
pip install -e ana_cont/
```

Expected output for the last command:

```
Successfully built ana_cont
Successfully installed ana_cont-x.x.x
```

### Verify the installation

```bash
python3 -c "import ana_cont.continuation as cont; print('ana_cont OK')"
```

If you see `ana_cont OK`, the library is installed correctly.

---

## Step 4 — Open the notebook

From inside `MaxEnt_benchmark/`, launch Jupyter:

```bash
jupyter lab benchmark_tutorial.ipynb
# or
jupyter notebook benchmark_tutorial.ipynb
```

When the notebook opens:

1. Click **Kernel → Change Kernel** (JupyterLab) or the kernel name in the
   top-right corner (VS Code / classic Notebook).
2. Select the environment where you ran `pip install` above (`.venv` or
   `maxent_bench`).
3. Run **Cell 1** (the imports cell). If it prints `Environment ready.`
   with no errors, everything is set up correctly.

---

## Step 5 — Provide your spectral function

Open the **Configuration** cell (the first code cell after the headers) and
set `YOUR_OMEGA` and `YOUR_A`:

```python
# Example: load from a CSV file
YOUR_OMEGA = np.linspace(-8, 8, 500)          # your omega grid
YOUR_A     = np.loadtxt('path/to/A_omega.csv') # 1-D array, same length
```

Everything else in the configuration cell is optional — leave it as `None`
if you do not have it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ana_cont` | Library not installed | Run `pip install -e ana_cont/` from inside `MaxEnt_benchmark/` |
| `ModuleNotFoundError: Cython` | Cython missing | Run `pip install cython` first, then reinstall ana_cont |
| `ModuleNotFoundError: torch` | PyTorch not installed | Run `pip install torch` (only needed for `.pt` files) |
| Kernel shows wrong Python version | Wrong kernel selected | Change the kernel to the environment where you ran `pip install` |
| `FileNotFoundError: Gbins_*.csv` | Data path is wrong | Check the path in the Configuration cell; the file must exist |
| `Environment ready.` not printed | Error in the imports cell | Read the full error message — it will name the missing package |
