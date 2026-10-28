"""Per-flow (graph-free) baselines on exactly the same split and features as the GNNs.

  xgboost  gradient-boosted trees (GPU, hist, early stopping on validation log-loss)
  rf       random forest via XGBRFClassifier (GPU)
  mlp      PyTorch MLP (GPU), early stopping on validation macro-F1
If xgboost is not installed, sklearn's HistGradientBoosting is used for xgboost/rf
(only meant for CPU smoke tests; the log says so).
"""
import time

import numpy as np

from .metrics import classification_metrics


def _sample_weight(y, power):
    if power <= 0:
        return None
    cnt = np.bincount(y)
    w = np.zeros(len(cnt))
    m = cnt > 0
    w[m] = (cnt[m].sum() / (m.sum() * cnt[m])) ** power
    w[m] /= w[m].mean()
    return w[y].astype(np.float32)


def _remap(y_train):
    present = np.unique(y_train)
    lut = -np.ones(int(present.max()) + 1, np.int64)
    lut[present] = np.arange(len(present))
    return present, lut


def _expand(prob_present, present, C):
    p = np.zeros((len(prob_present), C), np.float32)
    p[:, present] = prob_present
    return p


def fit_predict(kind, params, Xtr, ytr, Xva, yva, Xte, C, seed, threads, log=print):
    """Returns (test probabilities [n, C], fitted model, info)."""
    t0 = time.time()
    present, lut = _remap(ytr)
    ytr_r = lut[ytr]
    va_mask = np.isin(yva, present)
    Xva_r, yva_r = Xva[va_mask], lut[yva[va_mask]]
    sw = _sample_weight(ytr_r, float(params.get("weight_power", 0.0)))
    info = {}
    if kind in ("xgboost", "rf"):
        try:
            import xgboost as xgb
            have = True
        except ImportError:
            have = False
        if have:
            dev = "cuda" if _has_cuda() else "cpu"
            common = dict(tree_method="hist", device=dev, n_jobs=threads, random_state=seed,
                          max_bin=int(params.get("max_bin", 256)))
            if kind == "xgboost":
                clf = xgb.XGBClassifier(
                    n_estimators=int(params.get("n_estimators", 2000)), learning_rate=float(params.get("lr", 0.1)),
                    max_depth=int(params.get("max_depth", 8)), min_child_weight=float(params.get("min_child_weight", 1)),
                    subsample=float(params.get("subsample", 0.8)), colsample_bytree=float(params.get("colsample", 0.8)),
                    reg_lambda=float(params.get("reg_lambda", 1.0)), early_stopping_rounds=50,
                    eval_metric="mlogloss" if len(present) > 2 else "logloss", **common)
                clf.fit(Xtr, ytr_r, sample_weight=sw, eval_set=[(Xva_r, yva_r)], verbose=False)
                info["best_iteration"] = int(getattr(clf, "best_iteration", -1) or -1)
            else:
                clf = xgb.XGBRFClassifier(
                    n_estimators=int(params.get("n_estimators", 300)), max_depth=int(params.get("max_depth", 16)),
                    subsample=float(params.get("subsample", 0.8)), colsample_bynode=float(params.get("colsample", 0.8)),
                    min_child_weight=float(params.get("min_child_weight", 1)), **common)
                clf.fit(Xtr, ytr_r, sample_weight=sw)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            log("  [tabular] xgboost not installed -> sklearn HistGradientBoosting (smoke-test fallback)")
            clf = HistGradientBoostingClassifier(max_iter=int(params.get("n_estimators", 200)) if kind == "xgboost" else 100,
                                                 learning_rate=float(params.get("lr", 0.1)),
                                                 max_depth=int(params.get("max_depth", 8)), random_state=seed)
            clf.fit(Xtr, ytr_r, sample_weight=sw)
        pte = clf.predict_proba(Xte)
        pva = clf.predict_proba(Xva)
    elif kind == "mlp":
        clf, pva, pte = _mlp(params, Xtr, ytr_r, Xva_r, yva_r, Xva, Xte, len(present), seed, sw, log)
    else:
        raise ValueError(kind)
    info["fit_sec"] = time.time() - t0
    return _expand(pte, present, C), _expand(pva, present, C), clf, info


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _mlp(params, Xtr, ytr, Xva_r, yva_r, Xva, Xte, C, seed, sw, log):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, L, p = int(params.get("hidden", 256)), int(params.get("layers", 3)), float(params.get("dropout", 0.2))
    layers, d = [], Xtr.shape[1]
    for _ in range(L):
        layers += [nn.Linear(d, H), nn.BatchNorm1d(H), nn.ReLU(), nn.Dropout(p)]
        d = H
    net = nn.Sequential(*layers, nn.Linear(d, C)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(params.get("lr", 1e-3)), weight_decay=1e-5)
    w = None
    if sw is not None:
        cw = np.zeros(C, np.float32)
        for c in range(C):
            m = ytr == c
            cw[c] = sw[m][0] if m.any() else 0.0
        w = torch.tensor(cw, device=dev)
    Xt = torch.from_numpy(np.ascontiguousarray(Xtr)).to(dev)
    yt = torch.from_numpy(ytr).to(dev)
    bs = int(params.get("batch_size", 8192))
    epochs, patience = int(params.get("epochs", 30)), int(params.get("patience", 5))
    steps = int(params.get("steps_per_epoch", 2000))

    def infer(X):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 65536])).to(dev)
                out.append(torch.softmax(net(xb), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, C), np.float32)

    from sklearn.metrics import f1_score
    best, best_state, bad = -1, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        for _ in range(min(steps, max(1, len(Xt) // bs))):
            idx = torch.randint(0, len(Xt), (bs,), generator=g).to(dev)
            loss = F.cross_entropy(net(Xt[idx]), yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        f1 = f1_score(yva_r, infer(Xva_r).argmax(1), average="macro") if len(yva_r) else -loss.item()
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, infer(Xva), infer(Xte)


def run_tabular(cfg, data, split, kind, params, seed, log=print, model_path=None):
    store = data.make_store(split.graph_train, cfg["data"].get("ports", True), seed=0)
    Xtr, ytr = store.gather(split.train), data.y[split.train]
    va, te = np.sort(split.val), np.sort(split.test)
    Xva, yva = store.gather(va), data.y[va]
    Xte, yte = store.gather(te), data.y[te]
    threads = int(cfg["runtime"].get("threads", 4))
    pte, pva, clf, info = fit_predict(kind, params, Xtr, ytr, Xva, yva, Xte, data.n_classes, seed, threads, log)
    res = {"metrics": classification_metrics(yte, pte, data.classes),
           "val_macro_f1": classification_metrics(yva, pva, data.classes)["macro_f1"] if len(yva) else None,
           "info": info, "n_train": int(len(ytr)), "n_test": int(len(yte)), "classes": data.classes,
           "features": data.feature_names(store.ports)}
    if model_path and kind in ("xgboost", "rf") and hasattr(clf, "save_model"):
        clf.save_model(model_path)
    return res, clf, store
