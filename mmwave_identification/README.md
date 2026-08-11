# mmWave Person Identification

MSc dissertation artefact for identifying participants from millimetre-wave (mmWave) radar point clouds.

## 1. What is in this repository?

```text
mmwave-identification/
├── mmwave_identification.py        # original complete command-line pipeline
├── notebooks/
│   └── mmwave_identification.ipynb # supervisor-friendly notebook
├── mmwave_identification.ipynb     # easy-to-open GitHub copy
├── data/
│   └── raw/
│       └── README.md               # where the large dataset goes
├── results/
│   ├── classical/
│   └── figures/
├── requirements.txt                # CPU/classical + notebook dependencies
├── requirements-deep.txt           # adds PyTorch
└── README.md
```

**The large dataset is deliberately not uploaded to GitHub.** It should be copied to `data/raw/` on the computer where the experiment is run.

## 2. Technology stack

- Python 3.10–3.12
- NumPy — numerical arrays and preprocessing
- scikit-learn — classical ML models
- Matplotlib — figures
- pandas — notebook result tables
- Jupyter Notebook — interactive demonstration
- PyTorch — optional deep-learning models
- Git + GitHub — version control and code sharing

## 3. Dataset layout

The code expects:

```text
data/raw/
├── 0.pkl
├── 1.pkl
├── ...
├── 18.pkl
└── id.json
```

`id.json` maps participant IDs to session IDs. The `.pkl` files contain the radar point-cloud sessions.

Do **not** commit the dataset unless you have explicit permission and the repository is appropriate for storing it. The included `.gitignore` excludes the raw data.

## 4. Recommended setup

### Windows PowerShell

```powershell
git clone https://github.com/shajidhaque786/mmWave_point.git
cd mmwave-identification

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For deep models:

```powershell
pip install -r requirements-deep.txt
```

### macOS / Linux

```bash
git clone https://github.com/shajidhaque786/mmWave_point.git
cd mmwave-identification

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

For deep models:

```bash
pip install -r requirements-deep.txt
```

## 5. Put the dataset in place

Copy the supplied MiliPoint files into:

```text
data/raw/
```

Then verify that `data/raw/id.json` exists and that the session `.pkl` files referenced by it are present.

A quick check:

```bash
python -c "import json; print(json.load(open('data/raw/id.json')))"
```

## 6. Run the original Python pipeline

### Quick smoke test

Use a small frame limit first:

```bash
python mmwave_identification.py --all --frames 200
```

### Normal dissertation run

```bash
python mmwave_identification.py --all --frames 2000 --seeds 0 1 2
```

### Full available dataset

```bash
python mmwave_identification.py --all --frames 0 --seeds 0 1 2
```

This can be slow and memory-intensive. Only use it when the machine has been tested with the smaller run.

### Individual stages

```bash
python mmwave_identification.py --explore --frames 500
python mmwave_identification.py --classical --frames 500
python mmwave_identification.py --deep --frames 500
```

Useful options:

```text
--frames N          frames per session; 0 = all
--split random      published random split
--split block       leak-free block split
--split both        run both (recommended)
--seeds 0 1 2       repeat with multiple random seeds
--epochs 15         deep-model epochs
--batch 64          deep-model batch size
--class-weighted    class weighting for deep models
```

## 7. Run the notebook

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open:

```text
notebooks/mmwave_identification.ipynb
```

Run the cells from top to bottom.

### First supervisor demonstration

In the **Choose the experiment** cell use:

```python
frames=500
split="both"
seeds=[0]
epochs=5
deep=False
```

This demonstrates the data loading, exploration, feature engineering, classical models, leakage comparison and figures without waiting for the full deep-learning experiment.

After the pipeline is confirmed, increase the settings for the final run.

## 8. What the experiment compares

### Classical models

- Majority baseline
- Logistic Regression
- k-NN
- SVM (RBF)
- Random Forest
- Extra Trees
- Gradient Boosting
- MLP

### Deep models (PyTorch)

- TemporalPointNet — frame-aware bidirectional GRU
- TemporalPointNet-attn — self-attention over frames
- TemporalPointNet-mean — order-blind ablation
- PointNetLite — permutation-invariant reference

## 9. Why there are two evaluation protocols

This is a key methodological point in the project.

### Random split

All windows are shuffled before splitting. Because each sample contains 5 consecutive frames and neighbouring windows overlap, very similar windows can appear in both training and test sets. This can inflate accuracy.

### Block split

Windows are divided into contiguous time blocks within each session, with guard bands. This prevents training and test windows from sharing frames.

The notebook and Python script report both protocols side by side.

## 10. Metrics

Do not rely on accuracy alone because participant classes are imbalanced.

The pipeline reports:

- Top-1 accuracy
- Top-3 accuracy
- Balanced accuracy
- Macro-F1
- Weighted-F1
- Per-participant precision/recall/F1
- Confusion matrix
- Most frequent confusions

## 11. Outputs

After a successful run, look in:

```text
results/classical/results.csv
results/classical/summary.csv
results/figures/
```

The figures include model comparison, leakage gap, confusion matrices, per-class recall and feature importance.

## 12. How to present it to a supervisor

Use this order:

1. **Open the GitHub repository** and briefly show the project structure.
2. **Open the notebook** and explain that it is the reproducible front-end to the original `.py` implementation.
3. **Show the data folder structure** but explain that the raw dataset is excluded because it is too large / should remain local.
4. Run the environment-check cell.
5. Set `frames=200` or `500` for a quick live demonstration.
6. Load the data and show the tensor shape and participant count.
7. Show the sparsity/class-balance figures.
8. Explain the 5-frame stacking and the handcrafted features.
9. Run the classical models.
10. Show **random vs block** results and explain leakage.
11. Show balanced accuracy and macro-F1, not just top-1 accuracy.
12. If time permits, enable PyTorch and run a small deep-model demonstration.
13. Finish by showing where `summary.csv` and the figures are saved.
14. Explain that the final dissertation numbers are produced with the larger frame count and multiple seeds.

A concise explanation for the central contribution:

> “The important methodological issue is that consecutive radar windows overlap. A random split can therefore put near-identical observations into training and testing. I report the published random protocol for comparison, but I also introduce a block split with guard bands to estimate generalisation without frame overlap. I report both so the effect of evaluation leakage is visible rather than hidden.”

## 13. GitHub upload

Create a **private repository** first if the dissertation/data/code is not ready to be public.

From the project folder:

```bash
git init
git add .
git status
git commit -m "Initial mmWave identification pipeline and notebook"
git branch -M main
git remote add origin https://github.com/shajidhaque786/mmWave_point.git
git push -u origin main
```

Before the first push, check:

```bash
git status
```

You should **not** see the large `data/raw/*.pkl` files.

If you accidentally staged them:

```bash
git restore --staged data/raw
```

The `.gitignore` is included to prevent the raw dataset and generated results from being committed.

## 14. Reproducibility checklist

Before the final supervisor meeting:

- [ ] Clone the repository into a fresh folder.
- [ ] Create a fresh virtual environment.
- [ ] Install `requirements.txt`.
- [ ] Copy the dataset into `data/raw/`.
- [ ] Run the 200/500-frame smoke test.
- [ ] Run the notebook top-to-bottom without errors.
- [ ] Run the final `.py` command with the dissertation settings.
- [ ] Check `results/classical/summary.csv`.
- [ ] Check the generated figures.
- [ ] Confirm the GitHub repository does not contain the raw dataset.
- [ ] Keep the exact final command and software versions in your dissertation notes.

## 15. Important note about results

The README intentionally does **not** hard-code final accuracy numbers. Those should be generated from the dataset and experiment configuration actually used for the dissertation, then recorded with the corresponding seed/frame/epoch settings.
