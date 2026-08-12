# mmWave Person Identification

This project investigates **person identification using millimetre-wave (mmWave) radar point clouds**. It implements and evaluates classical machine-learning and deep-learning approaches for identifying participants from radar observations.

The project contains both:

* A complete Python implementation: `mmwave_identification.py`
* An interactive Jupyter Notebook: `mmwave_identification.ipynb`

The notebook is intended to make the complete experiment easy to understand, reproduce and demonstrate to a supervisor.

---
The required for this project is to large to be uploaded on github so here's the external link "https://drive.google.com/file/d/1rq8yyokrNhAGQryx7trpUqKenDnTI6Ky/view"
## 1. Project Structure

```text
mmWave_point/
│
├── README.md
│
├── mmwave_identification/
│   ├── mmwave_identification.py
│   ├── mmwave_identification.ipynb
│   └── notebooks/
│
├── data/
│   └── raw/
│       └── README.md
│
└── results/
```

The raw dataset is **not stored in GitHub** because of its large size.

---

# 2. Dataset

This project uses the **MiliPoint mmWave radar point-cloud dataset**.

The dataset contains radar point-cloud observations from multiple participants and sessions.

The raw dataset should be stored locally on the computer where the experiment is being run.

## Required dataset location

The code expects the data to be available at:

```text
data/raw/
```

The local project should therefore look like:

```text
mmWave_point/
│
├── README.md
│
├── mmwave_identification/
│   ├── mmwave_identification.py
│   └── mmwave_identification.ipynb
│
├── data/
│   └── raw/
│       ├── 0.pkl
│       ├── 1.pkl
│       ├── 2.pkl
│       ├── ...
│       ├── 18.pkl
│       └── id.json
│
└── results/
```

### Important

The `.pkl` files are the large raw dataset files and **should not be uploaded to GitHub**.

Only the dataset instructions are stored in the repository.

This means that anyone reproducing the experiment needs to obtain the dataset separately and place it inside:

```text
data/raw/
```

before running the code.

---

# 3. Dataset Files

The expected dataset contains session files such as:

```text
0.pkl
1.pkl
2.pkl
...
18.pkl
```

and the participant/session mapping file:

```text
id.json
```

The exact dataset files should match the version of the MiliPoint dataset used for the experiment.

---

# 4. Required Technology

The project uses the following technologies.

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python 3.10–3.12 | Main programming language                |
| NumPy            | Numerical processing and arrays          |
| scikit-learn     | Classical machine-learning models        |
| Matplotlib       | Data visualisation and figures           |
| pandas           | Results tables and analysis              |
| Jupyter Notebook | Interactive experiment and demonstration |
| PyTorch          | Deep-learning models                     |
| Git/GitHub       | Version control and project sharing      |

---

# 5. Python Environment

It is recommended to use a virtual environment.

## macOS / Linux

Create the environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

## Windows

Create the environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

# 6. Install Required Packages

Install the main dependencies:

```bash
pip install numpy scikit-learn matplotlib pandas jupyter ipykernel
```

For the deep-learning experiments, install PyTorch:

```bash
pip install torch
```

Alternatively, if `requirements.txt` is provided in the repository:

```bash
pip install -r requirements.txt
```

For the complete deep-learning environment:

```bash
pip install -r requirements-deep.txt
```

---

# 7. Running the Python Code

The main implementation is:

```text
mmwave_identification/mmwave_identification.py
```

The program provides several experiment modes.

## Quick test

For a fast demonstration:

```bash
python mmwave_identification/mmwave_identification.py --all --frames 200
```

## Medium demonstration

For a more representative run:

```bash
python mmwave_identification/mmwave_identification.py --all --frames 500
```

## Dissertation experiment

For the larger experiment:

```bash
python mmwave_identification/mmwave_identification.py --all --frames 2000 --seeds 0 1 2
```

The full available dataset can be used with:

```bash
python mmwave_identification/mmwave_identification.py --all --frames 0 --seeds 0 1 2
```

The full experiment may require significantly more memory and computation time.

---

# 8. Running Individual Experiments

The main stages can also be run separately.

### Dataset exploration

```bash
python mmwave_identification/mmwave_identification.py --explore --frames 500
```

### Classical machine learning

```bash
python mmwave_identification/mmwave_identification.py --classical --frames 500
```

### Deep learning

```bash
python mmwave_identification/mmwave_identification.py --deep --frames 500
```

---

# 9. Running the Jupyter Notebook

The notebook provides an interactive version of the experiment.

Open:

```text
mmwave_identification/mmwave_identification.ipynb
```

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open the notebook in the browser.

Run the cells from top to bottom.

---

# 10. Recommended Supervisor Demonstration

For a live demonstration, it is better to use a smaller dataset configuration first so that the complete pipeline can be shown without a long waiting time.

A recommended demonstration configuration is:

```text
Frames: 500
Split: both
Seeds: 0
Epochs: 5
Deep models: optional
```

The demonstration follows this order:

### 1. Environment

Show that Python and the required libraries are installed.

### 2. Dataset

Load the radar sessions and show:

* Number of samples
* Number of participants
* Number of sessions
* Point-cloud dimensions

### 3. Preprocessing

Explain that consecutive radar frames are stacked to capture temporal information.

### 4. Data exploration

Show the point-cloud and class-distribution figures.

### 5. Feature extraction

Explain the handcrafted features used by the classical models.

### 6. Classical models

Run the baseline machine-learning models.

### 7. Deep-learning models

Run the deep models if sufficient time and computational resources are available.

### 8. Evaluation protocols

Compare:

```text
Random Split
     vs
Block Split
```

This is an important methodological part of the project.

---

# 11. Evaluation Protocols

The project evaluates two different train/test splitting strategies.

## Random Split

Samples are randomly divided into training, validation and testing sets.

The problem is that consecutive radar windows can overlap.

Therefore, highly similar observations may appear in both training and testing.

This can result in an overly optimistic performance estimate.

## Block Split

The block split divides observations into contiguous temporal blocks and uses guard bands between partitions.

This reduces the possibility of overlapping or neighbouring observations appearing in both training and testing.

The project reports both protocols so that the effect of the evaluation strategy can be measured.

---

# 12. Machine-Learning Models

## Classical Models

The project evaluates classical machine-learning baselines including:

* Majority baseline
* Logistic Regression
* k-Nearest Neighbours
* Support Vector Machine
* Random Forest
* Extra Trees
* Gradient Boosting
* Multi-Layer Perceptron

## Deep-Learning Models

The project also includes point-cloud and temporal deep-learning approaches implemented using PyTorch.

These models investigate whether temporal information and learned point-cloud representations improve person identification.

---

# 13. Evaluation Metrics

The project reports several metrics rather than relying only on accuracy.

### Top-1 Accuracy

The percentage of samples where the predicted participant is the correct participant.

### Top-3 Accuracy

The percentage of samples where the correct participant appears among the three highest-ranked predictions.

### Balanced Accuracy

Useful when participant classes are not perfectly balanced.

### Macro-F1

Calculates F1 independently for each participant and then averages the results.

### Additional Metrics

The pipeline also produces:

* Weighted-F1
* Precision
* Recall
* Per-participant F1
* Confusion matrices
* Per-class performance

---

# 14. Results

Experiment outputs are stored under:

```text
results/
```

Typical outputs include:

```text
results/
├── classical/
│   ├── results.csv
│   └── summary.csv
│
└── figures/
    ├── model comparison
    ├── leakage comparison
    ├── confusion matrices
    └── per-class results
```

The exact files depend on the experiment configuration.

---

# 15. Main Research Question

A key methodological question investigated by this project is whether random splitting of overlapping temporal windows produces an overly optimistic estimate of person-identification performance.

The experiment therefore compares the standard random evaluation protocol with a block-based, leakage-resistant evaluation protocol.

The important comparison is:

```text
Random / overlapping split
            ↓
       Performance
            ↓
Block / guard-band split
            ↓
       Performance
```

This allows the effect of temporal overlap and evaluation leakage to be quantified.

---

# 16. Reproducibility

To reproduce the experiment:

1. Clone or download the repository.
2. Install Python.
3. Install the required packages.
4. Obtain the MiliPoint dataset.
5. Place the dataset inside `data/raw/`.
6. Run the small test configuration.
7. Run the notebook from beginning to end.
8. Run the final experiment using the documented frame count and seeds.
9. Save the generated results and figures.

The raw dataset is intentionally kept outside GitHub because of its size.

---

# 17. Recommended Final Experiment

For the final dissertation experiment, record the exact:

* Dataset version
* Number of frames
* Number of participants
* Number of sessions
* Random seeds
* Number of training epochs
* Batch size
* Evaluation protocol
* Python version
* PyTorch version
* scikit-learn version

This ensures that the final reported results can be reproduced.

---

# 18. Supervisor Explanation

A concise explanation of the project is:

> This project investigates person identification using mmWave radar point clouds. The pipeline preprocesses consecutive radar frames, extracts spatial and temporal information, and evaluates both classical machine-learning and deep-learning models. A particular focus is placed on the evaluation protocol because overlapping temporal windows can cause information leakage between training and testing. Therefore, the standard random split is compared with a block split using guard bands to obtain a more realistic estimate of model generalisation.

---

# 19. Dataset Location Summary

### On GitHub

```text
data/
└── raw/
    └── README.md
```

### On the local computer running the experiment

```text
data/
└── raw/
    ├── 0.pkl
    ├── 1.pkl
    ├── ...
    ├── 18.pkl
    └── id.json
```

**Do not upload the large `.pkl` files to GitHub.**

---

# 20. Project Goal

The overall goal is to determine how accurately individuals can be identified from mmWave radar point-cloud data and to determine how the evaluation methodology affects the reported performance.

The final results should therefore be interpreted together with the evaluation protocol used to obtain them.
