"""Training / evaluation of edge-classification GNNs on any split.

Evaluation is ALWAYS reported on 100% of the requested targets ("full"): targets that the
sampler did not put into the computation graph are scored with the targets injected
(force=True), so every model is compared on exactly the same test set. For samplers with
force=False (ladder step 0 = Multi-GNN behaviour) we additionally report the
Multi-GNN-style numbers computed only on the targets that happened to be sampled
("covered"), together with the seed coverage, because that is what the original
pipeline measures.
"""
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import graph_for
from .hardware import PowerMonitor, gpu_info, peak_vram_gb, reset_peak
from .loader import SubgraphBatches, make_loader, to_device
from .metrics import classification_metrics
from .models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_perf(threads=None):
    if threads:
        torch.set_num_threads(int(threads))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def autocast(enabled, device):
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def class_weights(counts, kind, power=0.5):
    counts = np.asarray(counts, np.float64)
    if kind == "ce":
        return None
    w = np.zeros_like(counts)
    m = counts > 0
    w[m] = (counts[m].sum() / (m.sum() * counts[m])) ** power
    w[m] /= w[m].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, graph, store, labels, targets, mode, cfg, seed, max_batches=None):
    """Probabilities for sorted `targets`; rows of targets absent from the computation graph
    are NaN. Returns (probs, coverage, twin_coverage)."""
    model.eval()
    device = next(model.parameters()).device
    t = cfg["train"]
    targets = np.sort(np.asarray(targets, np.int64))
    ds = SubgraphBatches(graph, store, labels, targets, t["eval_batch_size"], mode, cfg["sampler"],
                         shuffle=False, seed=seed + 1000, max_batches=max_batches)
    C = model.cls[-1].out_features
    probs = np.full((len(targets), C), np.nan, np.float32)
    pos, twins = 0, []
    for b in make_loader(ds, t["num_workers"]):
        n = len(b["targets"])
        present = b["present"].numpy()
        twins.append(b["twin"])
        if present.any():
            b = to_device(b, device)
            with autocast(t["amp"], device):
                out = model(b)
            p = torch.softmax(out["logits"].float(), 1).cpu().numpy()
            idx = pos + np.flatnonzero(present)
            probs[idx] = p
        pos += n
    covered = ~np.isnan(probs[:, 0])
    return probs, float(covered.mean()) if len(covered) else 0.0, float(np.mean(twins)) if twins else 0.0


def _metrics(y, probs, classes, ece_bins=15):
    ok = ~np.isnan(probs[:, 0])
    if not ok.any():
        return None
    return classification_metrics(y[ok], probs[ok], classes, ece_bins)


def mode_of(mcfg):
    return {"coupled": bool(mcfg["coupled"]), "force": bool(mcfg["force"])}


def train_and_evaluate(cfg, data, split, seed, log=print, ckpt_path=None, profile_steps=None, evaluate_test=True):
    """Train on split.train with message passing over split.graph_train, select the epoch on
    split.val (full coverage), test on split.test over split.graph_test."""
    set_seed(seed)
    device = get_device()
    m, s_cfg, t = cfg["model"], cfg["sampler"], cfg["train"]
    mode = mode_of(m)
    eval_mode = {"coupled": mode["coupled"], "force": True}
    store = data.make_store(split.graph_train, cfg["data"].get("ports", True), seed=0)
    g_train = graph_for(data, split, "train")
    counts = np.bincount(data.y[split.train], minlength=data.n_classes)
    log(f"  targets: train {len(split.train):,} | val {len(split.val):,} | test {len(split.test):,} | "
        f"train graph {g_train.n_edges:,} edges | mode {mode} | arch {m['arch']}")

    model = build_model(m, store.dim, data.n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    w = class_weights(counts, t["loss"], float(t.get("loss_power", 0.5)))
    w = w.to(device) if w is not None else None

    curve, best, best_f1, bad, step = [], None, -1.0, 0, 0
    seen = present_n = 0
    reset_peak()
    epoch_times = []
    max_steps = profile_steps or t.get("max_steps_per_epoch")
    with PowerMonitor() as pm:
        for epoch in range(int(t["epochs"])):
            model.train()
            ds = SubgraphBatches(g_train, store, data.y, split.train, int(t["batch_size"]), mode, s_cfg,
                                 shuffle=True, seed=seed, epoch=epoch, max_batches=max_steps)
            t0, losses = time.time(), []
            for b in make_loader(ds, t["num_workers"]):
                seen += len(b["targets"])
                present_n += int(b["present"].sum())
                if len(b["y"]) == 0:   # no target was sampled (possible with force=False)
                    continue
                b = to_device(b, device)
                with autocast(t["amp"], device):
                    out = model(b)
                loss = F.cross_entropy(out["logits"].float(), b["y"], weight=w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip"]))
                opt.step()
                step += 1
                losses.append(loss.item())
                if step % int(t["log_every"]) == 0:
                    curve.append([step, float(np.mean(losses[-int(t["log_every"]):]))])
            epoch_times.append(time.time() - t0)
            if profile_steps:
                break
            pv, _, _ = predict(model, g_train, store, data.y, split.val, eval_mode, cfg, seed)
            vm = _metrics(data.y[np.sort(split.val)], pv, data.classes)
            f1 = vm["macro_f1"] if vm else -1.0
            log(f"  epoch {epoch + 1:3d} | loss {np.mean(losses) if losses else float('nan'):.4f} | "
                f"val macro-F1 {f1:.4f} | train seed coverage {present_n / max(seen, 1):.3f} | "
                f"{epoch_times[-1]:.0f}s | VRAM {peak_vram_gb() or 0:.1f} GB")
            if f1 > best_f1 + 1e-4:
                best_f1, bad, best = f1, 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= int(t["patience"]):
                    log(f"  early stop at epoch {epoch + 1}")
                    break
        hw = {"peak_vram_gb": peak_vram_gb(), **pm.summary(), **gpu_info(),
              "sec_per_step": float(sum(epoch_times) / max(step, 1)), "steps": step,
              "epochs_run": len(epoch_times),
              "steps_per_full_epoch": int(math.ceil(len(split.train) / int(t["batch_size"])))}
    hw["targets_per_sec"] = float(int(t["batch_size"]) / hw["sec_per_step"]) if hw["sec_per_step"] else None
    if profile_steps:
        return {"hardware": hw, "train_seed_coverage": present_n / max(seen, 1)}, model, store

    if best is not None:
        model.load_state_dict(best)
    if not evaluate_test:
        return {"val_best_macro_f1": best_f1, "hardware": hw, "curve": curve}, model, store
    g_test = graph_for(data, split, "test")
    yt = data.y[np.sort(split.test)]
    t_inf = time.time()
    pf, _, _ = predict(model, g_test, store, data.y, split.test, eval_mode, cfg, seed)
    infer_sec = time.time() - t_inf
    res = {"metrics": _metrics(yt, pf, data.classes), "val_best_macro_f1": best_f1,
           "train_seed_coverage": present_n / max(seen, 1), "curve": curve,
           "hardware": dict(hw, test_inference_sec=infer_sec,
                            test_flows_per_sec=len(split.test) / max(infer_sec, 1e-9)),
           "graph_train": g_train.stats(), "n_train": int(len(split.train)), "n_test": int(len(split.test)),
           "classes": data.classes}
    if not mode["force"] or not mode["coupled"]:
        pc, cov, twin = predict(model, g_test, store, data.y, split.test, mode, cfg, seed)
        res["metrics_covered"] = _metrics(yt, pc, data.classes)
        res["test_seed_coverage"] = cov
        res["test_twin_coverage"] = twin
    if ckpt_path:
        torch.save({"state_dict": model.state_dict(), "model_cfg": m, "d_edge": store.dim,
                    "n_classes": data.n_classes, "store": store.state()}, ckpt_path)
    return res, model, store


def load_checkpoint(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_cfg"], ck["d_edge"], ck["n_classes"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
