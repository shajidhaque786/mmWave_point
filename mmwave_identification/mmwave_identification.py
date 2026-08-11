#!/usr/bin/env python3
# =============================================================================
#  mmWave Person Identification — COMPLETE SINGLE-FILE IMPLEMENTATION
# =============================================================================
#  MSc dissertation artefact.
#  Identifying people from millimetre-wave radar point clouds (MiliPoint dataset).
#
#  Everything in one file: data loading, feature engineering, classical models,
#  deep models, both evaluation protocols, metrics and figures.
#
#  -------------------------------------------------------------------------
#  QUICK START
#  -------------------------------------------------------------------------
#      pip install numpy scikit-learn matplotlib          # torch is OPTIONAL
#      python mmwave_identification.py --all              # everything (~10 min)
#
#      python mmwave_identification.py --explore          # dataset stats + figures
#      python mmwave_identification.py --classical        # RF / GB / SVM / kNN / LR
#      python mmwave_identification.py --deep             # temporal nets (needs torch)
#      python mmwave_identification.py --frames 500       # quick test
#      python mmwave_identification.py --all --frames 0   # ALL 545k frames (slow)
#
#  Expects the dataset in ./data/raw/ :
#      0.pkl … 18.pkl      per-session radar point clouds
#      id.json             participant -> session mapping
#
#  -------------------------------------------------------------------------
#  MODELS IMPLEMENTED
#  -------------------------------------------------------------------------
#   Classical (handcrafted features, scikit-learn)
#      Majority baseline · Logistic Regression · k-NN · SVM (RBF)
#      Random Forest · Extra Trees · Gradient Boosting · MLP (neural net)
#   Deep (raw point clouds, PyTorch — optional)
#      TemporalPointNet      frame-aware, bidirectional GRU over frames  [OURS]
#      TemporalPointNet-attn self-attention over frames                  [OURS]
#      TemporalPointNet-mean order-blind ABLATION CONTROL                [OURS]
#      PointNetLite          permutation-invariant reference
#
#  -------------------------------------------------------------------------
#  THE TWO EVALUATION PROTOCOLS  (this is the project's key contribution)
#  -------------------------------------------------------------------------
#   random : shuffle all windows, then split.  ← the published protocol.
#            Consecutive windows OVERLAP by stacks-1 frames, so near-identical
#            samples land in both train and test. Accuracy is inflated.
#   block  : within each session, take contiguous time blocks with guard bands.
#            No training window shares any frame with a test window.
#
#   Measured gap on real data: ~26 accuracy points. Reported side by side.
# =============================================================================
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------- config
STACKS = 5           # s : consecutive frames stacked into one sample
MAX_POINTS = 22      # k : fixed points per frame (sub-sample / zero-pad)
N_CLASSES = 11       # MiliPoint participants
SPLIT_SEED = 20      # matches the project config
FPS = 24

RAW = "data/raw"
OUT_RES = "results/classical"
OUT_FIG = "results/figures"
PEOPLE = [f"P{i}" for i in range(N_CLASSES)]

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_stacked(frames_per_session=2000, stacks=STACKS, k=MAX_POINTS, seed=SPLIT_SEED):
    """
    Load raw radar frames and build stacked samples, reproducing the official
    MiliPoint loader exactly.

    For every frame: if it has more than k points, randomly sub-sample to k;
    if fewer, zero-pad up to k. Then concatenate `stacks` consecutive frames
    ALONG THE POINT AXIS -> (stacks*k, 3).

    NOTE the flattening. The official loader does NOT keep a separate time
    dimension, which is why permutation-invariant models cannot tell which
    frame a point came from. Restoring that structure is contribution #2.

    Returns
        X    (N, stacks*k, 3) float32
        y    (N,) int         participant id
        sess (N,) int         source session (for the leak-free split)
        pos  (N,) float       relative position within the session, 0..1
    """
    idpath = os.path.join(RAW, "id.json")
    if not os.path.exists(idpath):
        sys.exit(f"ERROR: {idpath} not found. Put the dataset in {RAW}/ "
                 "(see ARTEFACT_README.md).")
    idmap = json.load(open(idpath))
    rng = np.random.default_rng(seed)
    X, y, sess_id, pos = [], [], [], []

    for pid in sorted(idmap, key=int):
        for sess in idmap[pid]:
            path = os.path.join(RAW, f"{sess}.pkl")
            if not os.path.exists(path):
                sys.exit(f"ERROR: missing {path}")
            data = pickle.load(open(path, "rb"))
            if frames_per_session:
                data = data[:frames_per_session]

            fixed = []
            for fr in data:
                p = np.asarray(fr["x"], dtype=np.float32)
                if p.shape[0] >= k:
                    p = p[rng.choice(p.shape[0], k, replace=False)]
                else:
                    p = np.vstack([p, np.zeros((k - p.shape[0], 3), np.float32)])
                fixed.append(p)

            n_win = len(fixed) - stacks + 1
            for i in range(n_win):
                X.append(np.vstack(fixed[i:i + stacks]))
                y.append(int(pid))
                sess_id.append(sess)
                pos.append(i / max(n_win - 1, 1))

    return (np.asarray(X, dtype=np.float32), np.asarray(y),
            np.asarray(sess_id), np.asarray(pos, dtype=np.float32))


# =============================================================================
# 2. EVALUATION PROTOCOLS
# =============================================================================
def split_random(n, seed=SPLIT_SEED, tr=0.8, va=0.1):
    """Published protocol: shuffle every window, then cut. LEAKY (see header)."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(tr * n), int((tr + va) * n)
    return idx[:a], idx[a:b], idx[b:]


def split_block(pos, sess, tr=0.8, va=0.1, guard=0.01):
    """
    Leak-free: contiguous time blocks WITHIN each session, plus guard bands so
    no training window shares a frame with a test window.

    A session-wise split is not usable here: several participants (P3, P5, P6,
    P7, P9, P10) were recorded in only ONE session, so holding out whole
    sessions would delete those classes entirely. Blocking within sessions
    keeps all 11 participants in every partition.
    """
    tr_i, va_i, te_i = [], [], []
    for s_ in np.unique(sess):
        m = np.where(sess == s_)[0]
        p = pos[m]
        tr_i.append(m[p < tr - guard])
        va_i.append(m[(p > tr + guard) & (p < tr + va - guard)])
        te_i.append(m[p > tr + va + guard])
    return np.concatenate(tr_i), np.concatenate(va_i), np.concatenate(te_i)


# =============================================================================
# 3. METRICS
# =============================================================================
def confusion_matrix(y_true, y_pred, n_cls=N_CLASSES):
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).astype(int).ravel(),
                    np.asarray(y_pred).astype(int).ravel()):
        cm[t, p] += 1
    return cm


def per_class_report(cm):
    """
    Per-class precision / recall / F1 plus balanced accuracy and macro-F1.

    The published benchmark reports ONLY aggregate top-1 accuracy. Because the
    classes are imbalanced by 3.67x, that number is dominated by the
    well-represented participants and can hide near-total failure on the rare
    ones. This is contribution #1.
    """
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    predicted = cm.sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    total = cm.sum()
    return {
        "confusion_matrix": cm, "support": support,
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": float(tp.sum() / total) if total else np.nan,
        "balanced_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "weighted_f1": float(np.nansum(f1 * support) / support.sum()) if support.sum() else np.nan,
    }


def evaluate(y_true, y_pred, proba=None, n_cls=N_CLASSES):
    rep = per_class_report(confusion_matrix(y_true, y_pred, n_cls))
    if proba is not None and np.ndim(proba) == 2 and proba.shape[1] >= 3:
        top3 = np.argsort(-proba, axis=1)[:, :3]
        rep["top3_accuracy"] = float((top3 == np.asarray(y_true)[:, None]).any(1).mean())
    else:
        rep["top3_accuracy"] = np.nan
    return rep


def format_report(rep, names=None):
    n = rep["confusion_matrix"].shape[0]
    names = names or PEOPLE
    L = ["Per-class performance", "-" * 62,
         f"{'person':>8} {'support':>9} {'precision':>10} {'recall':>8} {'f1':>8}", "-" * 62]
    for i in range(n):
        L.append(f"{names[i]:>8} {int(rep['support'][i]):>9} "
                 f"{rep['precision'][i]:>10.3f} {rep['recall'][i]:>8.3f} {rep['f1'][i]:>8.3f}")
    L += ["-" * 62,
          f"{'top-1 accuracy':<26}{rep['accuracy']:.4f}",
          f"{'top-3 accuracy':<26}{rep['top3_accuracy']:.4f}",
          f"{'balanced accuracy':<26}{rep['balanced_accuracy']:.4f}",
          f"{'macro-F1':<26}{rep['macro_f1']:.4f}", ""]
    gap = rep["accuracy"] - rep["balanced_accuracy"]
    L.append(f"accuracy - balanced accuracy = {gap:+.4f}  "
             "(positive => weaker on under-represented participants)")
    return "\n".join(L)


def top_confusions(cm, k=5, names=None):
    names = names or PEOPLE
    n = cm.shape[0]
    pairs = [(cm[i, j], names[i], names[j])
             for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    return [{"count": int(c), "true": t, "pred": p} for c, t, p in pairs[:k]]


# =============================================================================
# 4. FEATURE ENGINEERING  (for the classical models)
# =============================================================================
# Random Forests / SVMs cannot consume an unordered point set, so we build a
# fixed-length descriptor. The feature set deliberately covers the SAME
# information the deep models see, so the comparison is fair:
#   * body geometry  - centroid, extent, spread  (size/shape of the person)
#   * sparsity       - number of real detections (a weak but genuine cue)
#   * motion / gait  - frame-to-frame centroid displacement (the identity signal)
_PER_FRAME = ["n_points", "cx", "cy", "cz", "sx", "sy", "sz",
              "rx", "ry", "rz", "r_mean", "r_std", "bbox_vol", "spread"]


def _frame_stats(pts):
    if pts.shape[0] == 0:                      # frames with ZERO detections exist
        return np.zeros(len(_PER_FRAME))
    c, s = pts.mean(0), pts.std(0)
    rng = pts.max(0) - pts.min(0)
    r = np.linalg.norm(pts, axis=1)
    return np.array([pts.shape[0], *c, *s, *rng, r.mean(), r.std(),
                     float(np.prod(np.maximum(rng, 1e-6))),
                     np.linalg.norm(pts - c, axis=1).mean()])


def _feature_names():
    names = [f"{f}_{a}" for f in _PER_FRAME for a in ("mean", "std", "min", "max")]
    names += ["disp_mean", "disp_std", "disp_max", "disp_total",
              "vx_mean", "vy_mean", "vz_mean", "vx_std", "vy_std", "vz_std",
              "accel_mean", "accel_std"]
    names += ["g_cx", "g_cy", "g_cz", "g_sx", "g_sy", "g_sz",
              "g_rx", "g_ry", "g_rz", "g_r_mean", "g_r_std",
              "g_spread", "g_n_total", "g_n_frames_empty"]
    return names


FEATURE_NAMES = _feature_names()


def feature_groups():
    g = {}
    for nm in FEATURE_NAMES:
        if nm.startswith(("disp", "vx", "vy", "vz", "accel")):
            g[nm] = "motion / gait"
        elif nm.startswith("n_points") or nm.startswith("g_n_"):
            g[nm] = "sparsity"
        elif nm.startswith("g_"):
            g[nm] = "global geometry"
        else:
            g[nm] = "per-frame geometry"
    return g


def extract_features(samples, stacks=STACKS, k=MAX_POINTS, verbose=True):
    """(N, stacks*k, 3) point clouds -> (N, 82) feature matrix."""
    samples = np.asarray(samples, dtype=np.float64)
    n = samples.shape[0]
    out = np.zeros((n, len(FEATURE_NAMES)))

    for i in range(n):
        if verbose and n > 5000 and i % 10000 == 0 and i:
            print(f"    {i:,}/{n:,}", end="\r")
        frames = samples[i].reshape(stacks, k, 3)

        pf, cen, empties = [], [], 0
        for f in range(stacks):
            real = frames[f][np.abs(frames[f]).sum(1) > 0]     # drop padding
            if real.shape[0] == 0:
                empties += 1
            pf.append(_frame_stats(real))
            cen.append(real.mean(0) if real.shape[0] else np.zeros(3))
        pf = np.stack(pf); cen = np.stack(cen)

        # aggregate per-frame stats over time (mean/std/min/max, interleaved)
        agg = np.stack([pf.mean(0), pf.std(0), pf.min(0), pf.max(0)], axis=1).ravel()

        # motion / gait
        d = np.diff(cen, axis=0)
        dist = np.linalg.norm(d, axis=1)
        accel = np.diff(dist) if dist.size > 1 else np.zeros(1)
        motion = np.array([dist.mean(), dist.std(), dist.max(), dist.sum(),
                           *d.mean(0), *d.std(0), accel.mean(), accel.std()])

        # global cloud
        allp = samples[i]
        real_all = allp[np.abs(allp).sum(1) > 0]
        if real_all.shape[0]:
            gc = real_all.mean(0)
            gr = np.linalg.norm(real_all, axis=1)
            glob = np.array([*gc, *real_all.std(0),
                             *(real_all.max(0) - real_all.min(0)),
                             gr.mean(), gr.std(),
                             np.linalg.norm(real_all - gc, axis=1).mean(),
                             real_all.shape[0], empties])
        else:
            glob = np.zeros(14)

        out[i] = np.concatenate([agg, motion, glob])

    if verbose and n > 5000:
        print(" " * 30, end="\r")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# 5. CLASSICAL MODELS
# =============================================================================
def build_classical(seed=0):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler
    return {
        "Majority baseline":   DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(sc(), LogisticRegression(max_iter=2000, random_state=seed)),
        "k-NN (k=5)":          make_pipeline(sc(), KNeighborsClassifier(5, n_jobs=-1)),
        "SVM (RBF)":           make_pipeline(sc(), SVC(C=10, gamma="scale", random_state=seed)),
        "MLP (sklearn)":       make_pipeline(sc(), MLPClassifier((256, 128), max_iter=400,
                                                                 random_state=seed)),
        "Gradient Boosting":   HistGradientBoostingClassifier(max_iter=200, random_state=seed),
        "Extra Trees":         ExtraTreesClassifier(300, n_jobs=-1, random_state=seed),
        "Random Forest":       RandomForestClassifier(300, n_jobs=-1, random_state=seed),
    }


# =============================================================================
# 6. DEEP MODELS  (PyTorch — optional)
# =============================================================================
if HAS_TORCH:

    class FrameEncoder(nn.Module):
        """
        Permutation-invariant encoder for ONE radar frame.

        Shared per-point MLP + masked max/mean pooling. Masking matters
        enormously here: the median frame has only ~6 real points against a
        budget of 22, so ~72% of every input tensor is zero padding. A padded
        point means ABSENT, not a reflection at the origin.
        """

        def __init__(self, in_dim=3, hidden=64, out_dim=256):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
                nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.ReLU(True),
                nn.Linear(hidden * 2, out_dim))
            self.fuse = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.ReLU(True))

        def forward(self, x):                       # (B*s, k, 3)
            bs, k, c = x.shape
            mask = (x.abs().sum(-1) > 0).float().unsqueeze(-1)
            h = self.mlp(x.reshape(bs * k, c)).reshape(bs, k, -1) * mask

            # Masked max-pool. A frame can be ENTIRELY padding, so we must zero
            # those explicitly: filling with finfo.min and relying on
            # nan_to_num(neginf=..) does NOT work, because finfo.min is finite
            # (-3.4e38) and overflows the next Linear to -inf -> NaN loss.
            h_max = h.masked_fill(mask == 0, torch.finfo(h.dtype).min).max(1).values
            empty = (mask.sum(1) == 0)
            h_max = torch.where(empty, torch.zeros_like(h_max), h_max)

            h_mean = h.sum(1) / mask.sum(1).clamp(min=1.0)
            return self.fuse(torch.cat([h_max, h_mean], -1))

    class TemporalPointNet(nn.Module):
        """
        CONTRIBUTION #2 — the frame-aware model.

        The loader flattens s frames into one bag of s*k points, so every
        permutation-invariant baseline is blind to temporal order. But identity
        in mmWave lives in GAIT, which *is* temporal order.

        This model reshapes (B, s*k, 3) -> (B, s, k, 3), encodes each frame
        independently (order-invariant WITHIN a frame — correct), then models
        the sequence with a bidirectional GRU (order-SENSITIVE ACROSS frames —
        also correct).

        temporal_mode='mean' is the ABLATION CONTROL: same capacity, temporal
        order destroyed. It should perform worse if the hypothesis holds — which
        makes this a test rather than a claim.
        """

        def __init__(self, num_classes=N_CLASSES, max_points=MAX_POINTS,
                     stacks=STACKS, feat=256, temporal_mode="gru", dropout=0.3):
            super().__init__()
            self.num_classes, self.max_points = num_classes, max_points
            self.stacks, self.temporal_mode = stacks, temporal_mode
            self.enc = FrameEncoder(3, 64, feat)

            if temporal_mode == "gru":
                self.temporal = nn.GRU(feat, feat // 2, batch_first=True, bidirectional=True)
            elif temporal_mode == "attention":
                self.pos = nn.Parameter(torch.randn(1, stacks, feat) * 0.02)
                self.temporal = nn.TransformerEncoderLayer(
                    feat, nhead=4, dim_feedforward=feat, dropout=dropout, batch_first=True)
            elif temporal_mode == "mean":
                self.temporal = None
            else:
                raise ValueError(temporal_mode)

            self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(True),
                                      nn.Dropout(dropout), nn.Linear(256, num_classes))

        def forward(self, x):                       # (B, s*k, 3)
            b = x.shape[0]
            s, k = self.stacks, self.max_points
            f = self.enc(x.reshape(b, s, k, 3).reshape(b * s, k, 3)).reshape(b, s, -1)
            if self.temporal_mode == "gru":
                g = self.temporal(f)[0].mean(1)
            elif self.temporal_mode == "attention":
                g = self.temporal(f + self.pos[:, :s]).mean(1)
            else:
                g = f.mean(1)                       # ablation: order-blind
            return self.head(g)

    class PointNetLite(nn.Module):
        """Reference: permutation-invariant over ALL points (like the baselines)."""

        def __init__(self, num_classes=N_CLASSES, feat=256, dropout=0.3):
            super().__init__()
            self.num_classes = num_classes
            self.enc = FrameEncoder(3, 64, feat)
            self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(True),
                                      nn.Dropout(dropout), nn.Linear(256, num_classes))

        def forward(self, x):
            return self.head(self.enc(x))

    DEEP_MODELS = {
        "PointNetLite":            lambda: PointNetLite(),
        "TemporalPointNet (OURS)": lambda: TemporalPointNet(temporal_mode="gru"),
        "TemporalPointNet-attn":   lambda: TemporalPointNet(temporal_mode="attention"),
        "TemporalPointNet-mean (ABLATION)": lambda: TemporalPointNet(temporal_mode="mean"),
    }

    def train_deep(model, Xtr, ytr, Xva, yva, epochs=15, bs=64, lr=1e-3,
                   class_weighted=False, device="cpu", verbose=True):
        """Plain PyTorch training loop — no Lightning, no MPS (it crashes)."""
        model = model.to(device)
        weight = None
        if class_weighted:
            cnt = np.bincount(ytr, minlength=N_CLASSES).astype(float)
            w = cnt.sum() / (N_CLASSES * np.maximum(cnt, 1))
            weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        lossfn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)

        Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.long)
        Xva_t = torch.tensor(Xva).to(device); yva_t = torch.tensor(yva, dtype=torch.long)
        best, best_state = -1.0, None

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(Xtr_t))
            tot = 0.0
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
                opt.zero_grad()
                loss = lossfn(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("NaN/inf loss — numerical problem in the model")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()

            model.eval()
            with torch.no_grad():
                acc = (model(Xva_t).argmax(1).cpu() == yva_t).float().mean().item()
            if acc > best:
                best = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"      epoch {ep+1:2d}/{epochs}  loss={tot/len(perm):.4f}  val_acc={acc:.4f}",
                      end="\r")
        if verbose:
            print(" " * 70, end="\r")
        if best_state:
            model.load_state_dict(best_state)
        return model

    @torch.no_grad()
    def predict_deep(model, X, device="cpu", bs=256):
        model.eval().to(device)
        out = []
        for i in range(0, len(X), bs):
            out.append(model(torch.tensor(X[i:i + bs]).to(device)).cpu().numpy())
        return np.concatenate(out)


# =============================================================================
# 7. FIGURES
# =============================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25})
    return plt


C_MAIN, C_ALT, C_WARN, C_MUT = "#2E5395", "#7BAFD4", "#C44E52", "#8C8C8C"


def _save(fig, name):
    os.makedirs(OUT_FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG}/{name}.{ext}", bbox_inches="tight")
    _plt().close(fig)
    print(f"    {OUT_FIG}/{name}.png")


def figures_dataset(X, y, sess):
    """Sparsity, class balance and example point clouds."""
    plt = _plt()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # real points per frame, recovered from the stacked samples
    frames = X.reshape(-1, STACKS, MAX_POINTS, 3)
    counts = (np.abs(frames).sum(-1) > 0).sum(-1).ravel()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    ax[0].hist(counts, bins=np.arange(0, counts.max() + 2) - .5, color=C_MAIN,
               edgecolor="white", linewidth=.4)
    ax[0].axvline(MAX_POINTS, color=C_WARN, ls="--", lw=1.6, label=f"max_points = {MAX_POINTS}")
    ax[0].set_xlabel("detected points per frame"); ax[0].set_ylabel("frames")
    ax[0].set_title("Point-count distribution"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].annotate(f"median {int(np.median(counts))}   max {counts.max()}",
                   xy=(.97, .8), xycoords="axes fraction", ha="right", fontsize=8, color=C_MUT)
    pad = np.clip(1 - counts / MAX_POINTS, 0, 1) * 100
    ax[1].hist(pad, bins=30, color=C_ALT, edgecolor="white", linewidth=.4)
    ax[1].axvline(pad.mean(), color=C_WARN, ls="--", lw=1.6, label=f"mean {pad.mean():.0f}% padding")
    ax[1].set_xlabel("share of the input tensor that is padding (%)")
    ax[1].set_ylabel("frames"); ax[1].set_title("Zero-padding per frame")
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Radar sparsity: sub-sampling never fires; every frame is padded", y=1.03)
    _save(fig, "fig01_sparsity")

    ids, cnt = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    bars = ax.bar([f"P{i}" for i in ids], cnt, color=C_MAIN, edgecolor="white", linewidth=.5)
    for b in bars:
        if b.get_height() == cnt.max(): b.set_color(C_ALT)
        if b.get_height() == cnt.min(): b.set_color(C_WARN)
    ax.set_xlabel("participant"); ax.set_ylabel("samples")
    ax.set_title(f"Class imbalance — ratio {cnt.max()/cnt.min():.2f}×")
    _save(fig, "fig02_class_balance")

    fig = plt.figure(figsize=(10, 2.8))
    for j, pid in enumerate(ids[:4]):
        s = X[np.where(y == pid)[0][0]]
        p = s[np.abs(s).sum(1) > 0]
        a = fig.add_subplot(1, 4, j + 1, projection="3d")
        a.scatter(p[:, 0], p[:, 1], p[:, 2], s=22, c=p[:, 2], cmap="viridis", depthshade=False)
        a.set_title(f"P{pid} — {len(p)} points", fontsize=8)
        a.tick_params(labelsize=5); a.grid(False)
    fig.suptitle("A person, as the radar sees them (5 stacked frames)", y=1.04)
    _save(fig, "fig03_point_clouds")


def figures_results(rows, preds):
    plt = _plt()
    protos = sorted({r["protocol"] for r in rows}, reverse=True)
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])

    def g(m, p, k="top1"):
        v = [r[k] for r in rows if r["model"] == m and r["protocol"] == p]
        return float(np.mean(v)) if v else np.nan

    def gs(m, p, k="top1"):
        v = [r[k] for r in rows if r["model"] == m and r["protocol"] == p]
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

    models.sort(key=lambda m: -g(m, protos[0]))

    yy = np.arange(len(models)); h = .38
    fig, ax = plt.subplots(figsize=(7.6, .46 * len(models) + 1.7))
    if len(protos) > 1:
        ax.barh(yy + h/2, [g(m, "random")*100 for m in models], h, color=C_ALT,
                xerr=[gs(m, "random")*100 for m in models], error_kw=dict(ecolor=C_MUT, lw=.8),
                label="random split (published protocol)")
        ax.barh(yy - h/2, [g(m, "block")*100 for m in models], h, color=C_MAIN,
                xerr=[gs(m, "block")*100 for m in models], error_kw=dict(ecolor=C_MUT, lw=.8),
                label="block split (leak-free)")
    else:
        ax.barh(yy, [g(m, protos[0])*100 for m in models], h*2, color=C_MAIN)
    ax.axvline(100/N_CLASSES, color=C_WARN, ls=":", lw=1.4, label="chance (9.1%)")
    ax.set_yticks(yy); ax.set_yticklabels(models); ax.invert_yaxis()
    ax.set_xlabel("top-1 accuracy (%)"); ax.set_xlim(0, 100)
    ax.set_title("Identification accuracy — all models")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _save(fig, "fig04_model_comparison")

    if len(protos) > 1:
        ms = [m for m in models if m != "Majority baseline"]
        rnd = [g(m, "random")*100 for m in ms]; blk = [g(m, "block")*100 for m in ms]
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        for i, (r_, b_) in enumerate(zip(rnd, blk)):
            ax.plot([0, 1], [r_, b_], "-o", color=C_MAIN, ms=5, lw=1.3)
            ax.annotate(ms[i], xy=(1.03, b_), fontsize=7.5, va="center", color=C_MUT)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["random split\n(overlapping windows)", "block split\n(leak-free)"])
        ax.set_xlim(-.15, 1.6); ax.set_ylabel("top-1 accuracy (%)")
        ax.axhline(100/N_CLASSES, color=C_WARN, ls=":", lw=1.2)
        ax.set_title(f"Evaluation-protocol leakage: mean drop "
                     f"{np.mean(np.array(rnd)-np.array(blk)):.1f} points")
        _save(fig, "fig05_leakage")

    for proto in protos:
        keys = [k for k in preds if k.startswith(proto + "|")]
        if not keys:
            continue
        bk = max(keys, key=lambda k: preds[k]["rep"]["accuracy"])
        rep = preds[bk]["rep"]; name = bk.split("|", 1)[1]
        cm = rep["confusion_matrix"].astype(float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)

        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=.046, label="proportion of true class")
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(PEOPLE, rotation=45, ha="right")
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(PEOPLE)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
        ax.set_title(f"Confusion matrix — {name}\n({proto} split, acc {rep['accuracy']*100:.1f}%)",
                     fontsize=9)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if cmn[i, j] > .01:
                    ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if cmn[i, j] < .5 else "black")
        _save(fig, f"fig06_confusion_{proto}")

        rec, sup = rep["recall"], rep["support"]
        order = np.argsort(sup)
        fig, ax = plt.subplots(figsize=(6.6, 2.9))
        ax.bar(range(N_CLASSES), rec[order]*100, color=C_MAIN, edgecolor="white", linewidth=.5)
        ax.axhline(np.nanmean(rec)*100, color=C_WARN, ls="--", lw=1.3,
                   label=f"balanced accuracy {np.nanmean(rec)*100:.1f}%")
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels([f"{PEOPLE[i]}\n{int(sup[i])}" for i in order], fontsize=7)
        ax.set_xlabel("participant (ordered by test support, rarest first)")
        ax.set_ylabel("recall (%)"); ax.set_ylim(0, 100)
        ax.set_title(f"Per-class recall — {name} ({proto} split)")
        ax.legend(frameon=False, fontsize=8)
        _save(fig, f"fig07_per_class_recall_{proto}")

    imp = next((preds[k]["importances"] for k in preds if "importances" in preds[k]), None)
    if imp is not None:
        idx = np.argsort(imp)[::-1][:18]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh(range(len(idx)), imp[idx][::-1]*100, color=C_MAIN)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([FEATURE_NAMES[i] for i in idx][::-1], fontsize=7)
        ax.set_xlabel("importance (%)"); ax.set_title("Most informative handcrafted features")
        _save(fig, "fig08_feature_importance")

        grp = feature_groups(); agg = {}
        for i, nm in enumerate(FEATURE_NAMES):
            agg[grp[nm]] = agg.get(grp[nm], 0) + imp[i]
        ks = sorted(agg, key=agg.get, reverse=True)
        fig, ax = plt.subplots(figsize=(5.4, 2.8))
        ax.bar(ks, [agg[k]*100 for k in ks], color=[C_MAIN, C_ALT, C_MUT, C_WARN][:len(ks)],
               edgecolor="white")
        ax.set_ylabel("total importance (%)")
        ax.set_title("Which kind of information carries identity?")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
        _save(fig, "fig09_feature_groups")


# =============================================================================
# 8. EXPERIMENT DRIVER
# =============================================================================
def run(args):
    t0 = time.time()
    print("=" * 78)
    print(" mmWave PERSON IDENTIFICATION — complete pipeline")
    print("=" * 78)
    print(f"  stacks={STACKS}  max_points={MAX_POINTS}  ->  input ({STACKS*MAX_POINTS}, 3)")
    print(f"  torch available: {HAS_TORCH}")

    print(f"\n[1] Loading data (frames/session = {args.frames or 'ALL'}) ...")
    X, y, sess, pos = load_stacked(args.frames)
    print(f"    samples {X.shape}   participants {len(np.unique(y))}   "
          f"({time.time()-t0:.1f}s)")

    if args.explore or args.all:
        print("\n[2] Dataset figures ...")
        figures_dataset(X, y, sess)

    need_feats = args.classical or args.all
    if need_feats:
        print("\n[3] Extracting handcrafted features ...")
        t = time.time()
        F = extract_features(X)
        print(f"    features {F.shape}  ({time.time()-t:.1f}s)")

    protocols = ["random", "block"] if args.split == "both" else [args.split]
    rows, preds = [], {}

    for proto in protocols:
        print(f"\n{'='*78}\n PROTOCOL: {proto.upper()}"
              + ("   (published — overlapping windows, LEAKY)" if proto == "random"
                 else "   (leak-free — contiguous time blocks)"))
        print("=" * 78)
        tr, va, te = (split_random(len(X)) if proto == "random"
                      else split_block(pos, sess))
        print(f"  train {len(tr):,}   val {len(va):,}   test {len(te):,}")

        for seed in args.seeds:
            print(f"\n  --- seed {seed} ---")

            if need_feats:
                for name, model in build_classical(seed).items():
                    t = time.time()
                    model.fit(F[tr], y[tr])
                    yp = model.predict(F[te])
                    proba = model.predict_proba(F[te]) if hasattr(model, "predict_proba") else None
                    rep = evaluate(y[te], yp, proba)
                    rows.append(dict(model=name, protocol=proto, seed=seed,
                                     top1=rep["accuracy"], top3=rep["top3_accuracy"],
                                     balanced=rep["balanced_accuracy"],
                                     macro_f1=rep["macro_f1"], seconds=time.time()-t))
                    print(f"    {name:<34} top1={rep['accuracy']*100:6.2f}%  "
                          f"bal={rep['balanced_accuracy']*100:6.2f}%  "
                          f"F1={rep['macro_f1']*100:6.2f}%  ({time.time()-t:.1f}s)")
                    if seed == args.seeds[0]:
                        e = dict(rep=rep)
                        if hasattr(model, "feature_importances_"):
                            e["importances"] = model.feature_importances_
                        preds[f"{proto}|{name}"] = e

            if (args.deep or args.all) and HAS_TORCH:
                for name, ctor in DEEP_MODELS.items():
                    t = time.time()
                    torch.manual_seed(seed); np.random.seed(seed)
                    try:
                        m = train_deep(ctor(), X[tr], y[tr], X[va], y[va],
                                       epochs=args.epochs, bs=args.batch,
                                       class_weighted=args.class_weighted)
                        logits = predict_deep(m, X[te])
                        rep = evaluate(y[te], logits.argmax(1), logits)
                    except RuntimeError as e:
                        print(f"    {name:<34} FAILED: {e}")
                        continue
                    rows.append(dict(model=name, protocol=proto, seed=seed,
                                     top1=rep["accuracy"], top3=rep["top3_accuracy"],
                                     balanced=rep["balanced_accuracy"],
                                     macro_f1=rep["macro_f1"], seconds=time.time()-t))
                    print(f"    {name:<34} top1={rep['accuracy']*100:6.2f}%  "
                          f"bal={rep['balanced_accuracy']*100:6.2f}%  "
                          f"F1={rep['macro_f1']*100:6.2f}%  ({time.time()-t:.1f}s)")
                    if seed == args.seeds[0]:
                        preds[f"{proto}|{name}"] = dict(rep=rep)
            elif (args.deep or args.all) and not HAS_TORCH:
                print("    [deep models skipped — torch not installed]")

    if not rows:
        sys.exit("no results produced")

    # ---------------------------------------------------------------- report
    os.makedirs(OUT_RES, exist_ok=True)
    with open(f"{OUT_RES}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    protos = sorted({r["protocol"] for r in rows}, reverse=True)
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])

    def stat(m, p, k):
        v = [r[k] for r in rows if r["model"] == m and r["protocol"] == p]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) \
            if v else (np.nan, 0.0)

    models.sort(key=lambda m: -stat(m, protos[0], "top1")[0])

    with open(f"{OUT_RES}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "protocol", "n_seeds", "top1_mean", "top1_std",
                    "top3_mean", "balanced_mean", "macro_f1_mean"])
        for m in models:
            for p in protos:
                rs = [r for r in rows if r["model"] == m and r["protocol"] == p]
                if rs:
                    w.writerow([m, p, len(rs), *[stat(m, p, k)[0] for k in
                                                 ("top1",)], stat(m, p, "top1")[1],
                                stat(m, p, "top3")[0], stat(m, p, "balanced")[0],
                                stat(m, p, "macro_f1")[0]])

    print("\n" + "=" * 92)
    print(" FINAL RESULTS")
    print("=" * 92)
    if len(protos) > 1:
        print(f"{'model':<34}{'RANDOM (leaky)':>18}{'BLOCK (leak-free)':>20}{'gap':>12}")
        print("-" * 92)
        for m in models:
            a = stat(m, "random", "top1")[0]; b = stat(m, "block", "top1")[0]
            print(f"{m:<34}{a*100:16.2f}%{b*100:19.2f}%{(a-b)*100:+11.1f}")
        print("=" * 92)
        gaps = [stat(m, "random", "top1")[0] - stat(m, "block", "top1")[0]
                for m in models if m != "Majority baseline"]
        print(f"\n  Mean leakage gap: {np.mean(gaps)*100:+.1f} accuracy points.")
        print("  The random split shuffles OVERLAPPING windows, so near-duplicate")
        print("  samples appear in both train and test. The block split is the")
        print("  honest estimate of generalisation.")
    else:
        p = protos[0]
        print(f"{'model':<34}{'top-1':>12}{'balanced':>12}{'macro-F1':>12}")
        print("-" * 92)
        for m in models:
            print(f"{m:<34}{stat(m,p,'top1')[0]*100:11.2f}%"
                  f"{stat(m,p,'balanced')[0]*100:11.2f}%{stat(m,p,'macro_f1')[0]*100:11.2f}%")

    best = max(preds, key=lambda k: preds[k]["rep"]["accuracy"])
    print(f"\n  Best model: {best.split('|',1)[1]}  ({best.split('|')[0]} split)\n")
    print(format_report(preds[best]["rep"]))
    print("\n  Most frequent confusions:")
    for c in top_confusions(preds[best]["rep"]["confusion_matrix"]):
        print(f"    {c['true']} -> {c['pred']}: {c['count']}")

    print("\n[4] Result figures ...")
    figures_results(rows, preds)

    print(f"\n{'='*78}")
    print(f"  results : {OUT_RES}/summary.csv, results.csv")
    print(f"  figures : {OUT_FIG}/*.png (report) and *.pdf (print)")
    print(f"  total   : {time.time()-t0:.0f}s")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(
        description="mmWave person identification — complete single-file pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="everything (recommended)")
    ap.add_argument("--explore", action="store_true", help="dataset stats + figures only")
    ap.add_argument("--classical", action="store_true", help="classical ML models")
    ap.add_argument("--deep", action="store_true", help="deep models (needs torch)")
    ap.add_argument("--frames", type=int, default=2000,
                    help="frames per session (0 = all 545k; slow). default 2000")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--split", choices=["random", "block", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=15, help="deep-model epochs")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--class-weighted", action="store_true",
                    help="inverse-frequency class weighting for deep models")
    args = ap.parse_args()

    if not any([args.all, args.explore, args.classical, args.deep]):
        args.all = True
    run(args)


if __name__ == "__main__":
    main()
