"""Equal-budget hyper-parameter search (validation macro-F1 on split p1A).

Every model gets the same number of trials (tuning.trials). GNN trials use a shortened
schedule (tuning.gnn_epochs x tuning.gnn_steps); the test split is never touched.
Uses Optuna (TPE, fixed seed) when installed, otherwise random search with the same spaces.
Output: results/tuned/<model>__<dataset>.json with best_params (dotted config keys for GNNs,
plain parameter names for tabular models).
"""
import copy
import json
import os
import time

import numpy as np

from .data import NetFlowData, get_split
from .jobs import set_dotted


def gnn_space(model):
    sp = {
        "model.hidden": ("cat", [64, 128, 256]),
        "model.layers": ("cat", [2, 3]),
        "model.dropout": ("float", 0.0, 0.5),
        "train.lr": ("logfloat", 1e-4, 3e-3),
        "train.loss": ("cat", ["ce", "wce"]),
        "train.loss_power": ("float", 0.25, 1.0),
        "train.batch_size": ("cat", [2048, 4096, 8192]),
        "sampler.fanout": ("cat", [[10, 10], [20, 20], [50, 50]] + ([[100, 100]] if model == "multignn" else [])),
    }
    if model != "egsage":
        sp["model.conv"] = ("cat", ["gine", "gatv2"])
    return sp


TAB_SPACES = {
    "xgboost": {"lr": ("logfloat", 0.02, 0.3), "max_depth": ("int", 4, 12), "min_child_weight": ("logfloat", 1, 20),
                "subsample": ("float", 0.5, 1.0), "colsample": ("float", 0.5, 1.0), "reg_lambda": ("logfloat", 0.1, 10),
                "weight_power": ("float", 0.0, 1.0)},
    "rf": {"n_estimators": ("cat", [100, 300, 500]), "max_depth": ("int", 8, 20), "subsample": ("float", 0.5, 1.0),
           "colsample": ("float", 0.3, 1.0), "weight_power": ("float", 0.0, 1.0)},
    "mlp": {"hidden": ("cat", [128, 256, 512]), "layers": ("int", 2, 4), "dropout": ("float", 0.0, 0.5),
            "lr": ("logfloat", 1e-4, 3e-3), "weight_power": ("float", 0.0, 1.0)},
}


class _RandomTrial:
    def __init__(self, rng):
        self.rng, self.params = rng, {}

    def suggest(self, name, spec):
        kind = spec[0]
        if kind == "cat":
            v = spec[1][self.rng.integers(len(spec[1]))]
        elif kind == "int":
            v = int(self.rng.integers(spec[1], spec[2] + 1))
        elif kind == "float":
            v = float(self.rng.uniform(spec[1], spec[2]))
        else:
            v = float(np.exp(self.rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        self.params[name] = v
        return v


def _suggest_optuna(trial, name, spec):
    kind = spec[0]
    if kind == "cat":
        # optuna categorical choices must be primitives: encode lists as strings
        choices = [json.dumps(c) if isinstance(c, list) else c for c in spec[1]]
        v = trial.suggest_categorical(name, choices)
        return json.loads(v) if isinstance(v, str) and v.startswith("[") else v
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    return trial.suggest_float(name, spec[1], spec[2], log=True)


def tune(cfg, spec, root, log):
    from .executors import resolve, tabular_params
    model = spec["model"]
    kind = cfg["models"][model]["kind"]
    base = resolve(cfg, dict(spec, tuned=None, overrides={}), root)
    data = NetFlowData(base, spec["dataset"], task="multi", node_mode=base["_node_mode"])
    split = get_split(base, data, spec["split"])
    space = gnn_space(model) if kind == "gnn" else TAB_SPACES[model]
    history = []

    if kind == "tabular":
        from .tabular import fit_predict
        from .metrics import classification_metrics
        store = data.make_store(split.graph_train, base["data"].get("ports", True), seed=0)
        Xtr, ytr = store.gather(split.train), data.y[split.train]
        va = np.sort(split.val)
        Xva, yva = store.gather(va), data.y[va]

    def objective(sample):
        t0 = time.time()
        if kind == "gnn":
            from .trainer import train_and_evaluate
            c = copy.deepcopy(base)
            for k in space:
                set_dotted(c, k, sample(k, space[k]))
            c["train"].update(epochs=int(cfg["tuning"]["gnn_epochs"]), max_steps_per_epoch=int(cfg["tuning"]["gnn_steps"]),
                              patience=3)
            res, _, _ = train_and_evaluate(c, data, split, 0, log=lambda *_: None, evaluate_test=False)
            score = res["val_best_macro_f1"]
        else:
            p = tabular_params(base, model)
            for k in space:
                p[k] = sample(k, space[k])
            _, pva, _, _ = fit_predict(model, p, Xtr, ytr, Xva, yva, Xva[:1], data.n_classes, 0,
                                       int(base["runtime"].get("threads", 4)), log=lambda *_: None)
            score = classification_metrics(yva, pva, data.classes)["macro_f1"]
        history.append({"score": float(score), "sec": time.time() - t0})
        log(f"  [tune {model}/{spec['dataset']}] trial {len(history)}: val macro-F1 {score:.4f}")
        return float(score)

    n = int(spec.get("trials", cfg["tuning"]["trials"]))
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(lambda tr: objective(lambda k, s: _suggest_optuna(tr, k, s)), n_trials=n)
        best = {k: (json.loads(v) if isinstance(v, str) and v.startswith("[") else v)
                for k, v in study.best_params.items()}
        best_score, engine = study.best_value, "optuna-tpe"
    except ImportError:
        rng = np.random.default_rng(0)
        best, best_score = None, -1
        for _ in range(n):
            tr = _RandomTrial(rng)
            s = objective(tr.suggest)
            if s > best_score:
                best, best_score = dict(tr.params), s
        engine = "random"
    out = {"model": model, "dataset": spec["dataset"], "best_params": best, "best_val_macro_f1": best_score,
           "trials": history, "engine": engine}
    tmp = os.path.join(root, "tuned", f"{model}__{spec['dataset']}.json")
    with open(tmp + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp + ".tmp", tmp)
    return out
