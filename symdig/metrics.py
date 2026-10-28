"""Evaluation metrics (numpy / scikit-learn)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true, prob, classes, ece_bins=15):
    pred = prob.argmax(1)
    labels = np.arange(len(classes))
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    present = s > 0  # macro averages over classes present in the test split
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels[present], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "ece": float(expected_calibration_error(y_true, prob, ece_bins)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(classes)},
    }


def expected_calibration_error(y_true, prob, n_bins=15):
    """Top-label ECE with equal-width confidence bins (Guo et al., 2017)."""
    conf = prob.max(1)
    correct = (prob.argmax(1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return ece


def effective_rank(H):
    """Erank(H) = exp(-sum p_m log p_m), p_m = sigma_m / sum sigma (Roy & Vetterli)."""
    H = np.asarray(H, np.float64)
    s = np.linalg.svd(H, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def mean_pairwise_cosine(H, max_n=4096, seed=0):
    H = np.asarray(H, np.float64)
    if len(H) > max_n:
        H = H[np.random.default_rng(seed).choice(len(H), max_n, replace=False)]
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    S = Hn @ Hn.T
    n = len(Hn)
    return float((S.sum() - np.trace(S)) / max(n * (n - 1), 1))


def singular_spectrum(H):
    H = np.asarray(H, np.float64)
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    s = np.linalg.svd(Hn, compute_uv=False)
    return s / s[0] if len(s) and s[0] > 0 else s


def collapse_stats(H):
    return {"cos": mean_pairwise_cosine(H), "erank": effective_rank(H), "dim": int(np.asarray(H).shape[1]),
            "n_nodes": int(len(H))}
