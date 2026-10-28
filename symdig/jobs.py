"""File-based job queue shared by any number of workers / GPUs / pods.

results/queue/{pending,running,done,failed}/<id>.json   job specs (claimed by atomic rename)
results/runs/<id>.json                                   outputs (spec + resolved config + result)
results/index/<exp>.json                                 ids planned for each experiment

A job id is the hash of its spec without 'exp' / 'after' / 'priority', so the same run
requested by two experiments is executed once. Dependencies: a job is claimable only when
every id in spec['after'] has an output.
"""
import copy
import hashlib
import json
import os
import socket
import time

import yaml

STATES = ("pending", "running", "done", "failed")
HASH_EXCLUDE = ("exp", "after", "priority")


def load_config(path, overrides=()):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for o in overrides or ():
        k, v = o.split("=", 1)
        set_dotted(cfg, k, yaml.safe_load(v))
    return cfg


def set_dotted(cfg, key, value):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def job_id(spec):
    s = {k: v for k, v in spec.items() if k not in HASH_EXCLUDE}
    return hashlib.sha1(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Queue:
    def __init__(self, root):
        self.root = root
        for s in STATES:
            os.makedirs(os.path.join(root, "queue", s), exist_ok=True)
        for d in ("runs", "index", "ckpt", "logs", "tuned", "models"):
            os.makedirs(os.path.join(root, d), exist_ok=True)

    def p(self, state, jid):
        return os.path.join(self.root, "queue", state, f"{jid}.json")

    def out(self, jid):
        return os.path.join(self.root, "runs", f"{jid}.json")

    def has_output(self, jid):
        return os.path.exists(self.out(jid))

    def state_of(self, jid):
        for s in STATES:
            if os.path.exists(self.p(s, jid)):
                return s
        return "done" if self.has_output(jid) else None

    def add(self, spec):
        jid = job_id(spec)
        spec = dict(spec, id=jid)
        st = self.state_of(jid)
        if st in ("pending", "running") or self.has_output(jid):
            if st == "pending":   # merge experiment tags / dependencies
                old = json.load(open(self.p("pending", jid)))
                old["exps"] = sorted(set(old.get("exps", [])) | {spec.get("exp")})
                old["after"] = sorted(set(old.get("after", [])) | set(spec.get("after", [])))
                self._write(self.p("pending", jid), old)
            return jid, False
        if st == "failed":
            os.remove(self.p("failed", jid))
        spec["exps"] = [spec.get("exp")]
        self._write(self.p("pending", jid), spec)
        return jid, True

    def _write(self, path, obj):
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, default=str)
        os.replace(tmp, path)

    def list(self, state):
        d = os.path.join(self.root, "queue", state)
        return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))

    def spec(self, jid):
        for s in STATES:
            if os.path.exists(self.p(s, jid)):
                return json.load(open(self.p(s, jid)))
        if self.has_output(jid):
            return json.load(open(self.out(jid)))["spec"]
        raise KeyError(jid)

    def claim(self, accept=None):
        """Atomically claim the next runnable job (lowest priority first). Returns (spec, n_blocked)."""
        cands = []
        for jid in self.list("pending"):
            try:
                cands.append(json.load(open(self.p("pending", jid))))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        cands.sort(key=lambda s: (s.get("priority", 50), s["id"]))
        blocked = 0
        for s in cands:
            if accept and not accept(s):
                continue
            deps = s.get("after", [])
            if any(not self.has_output(d) for d in deps):
                blocked += 1
                continue
            try:
                os.rename(self.p("pending", s["id"]), self.p("running", s["id"]))
            except (FileNotFoundError, OSError):
                continue
            return s, blocked
        return None, blocked

    def failed_deps(self, spec):
        return [d for d in spec.get("after", []) if os.path.exists(self.p("failed", d))]

    def finish(self, spec, result, cfg, t0, ok=True, error=None):
        jid = spec["id"]
        rec = {"id": jid, "spec": spec, "cfg": cfg, "result": result, "started": t0, "finished": time.time(),
               "host": socket.gethostname(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if ok:
            self._write(self.out(jid), rec)
            dst = self.p("done", jid)
        else:
            rec["error"] = error
            dst = self.p("failed", jid)
        self._write(dst, dict(spec, error=error) if not ok else spec)
        if os.path.exists(self.p("running", jid)):
            os.remove(self.p("running", jid))

    def requeue(self, states=("running",)):
        n = 0
        for s in states:
            for jid in self.list(s):
                os.replace(self.p(s, jid), self.p("pending", jid))
                n += 1
        return n

    def record_index(self, exp, ids):
        f = os.path.join(self.root, "index", f"{exp}.json")
        old = json.load(open(f)) if os.path.exists(f) else []
        new = sorted(set(old) | set(ids))
        self._write(f, new)

    def runs_of(self, exp):
        f = os.path.join(self.root, "index", f"{exp}.json")
        if not os.path.exists(f):
            return []
        out = []
        for jid in json.load(open(f)):
            if self.has_output(jid):
                out.append(json.load(open(self.out(jid))))
        return out
