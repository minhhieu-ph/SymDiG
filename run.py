"""SymDiG research pipeline: plan experiments, run workers, inspect the queue.

  python run.py plan E1 E2 E3 E4 E5 E6 E7      # add jobs (dedup; safe to re-run)
  python launch.py                             # one worker per GPU (see launch.py --help)
  python run.py status                         # progress, failures, blocked jobs
  python run.py worker                         # a single worker on the current device
  python run.py run <job-id>                   # run one job in the foreground (debugging)
  python run.py requeue                        # move 'running' jobs of dead workers back to pending
  python run.py retry                          # move failed jobs back to pending
  python run.py smoke --datasets ton           # tiny end-to-end run of every experiment
  python make_tables.py && python make_figures.py

Options: --config configs/default.yaml  --results results  --set key=value ...
         plan: --datasets ton unsw ...  --seeds 0 1 2
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback

import yaml

from symdig.jobs import Queue, load_config

EXPS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "tune"]
CFG_HASH_KEYS = ("data", "protocols", "model", "sampler", "train", "ladder", "models", "multignn_paper",
                 "tabular_defaults", "fecograph", "tuning", "motif")


def cfg_hash(cfg):
    blob = json.dumps({k: cfg.get(k) for k in CFG_HASH_KEYS}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def saved_config_path(results):
    return os.path.join(results, "config.yaml")


def load_run_config(args):
    """Workers use the snapshot written at plan time, so a job always runs with the
    configuration it was planned with."""
    p = saved_config_path(args.results)
    if os.path.exists(p) and args.cmd not in ("plan", "smoke"):
        cfg = yaml.safe_load(open(p))
    else:
        cfg = load_config(args.config, args.set)
    cfg["paths"]["results"] = args.results
    cfg.setdefault("runtime", {})
    cfg["runtime"]["threads"] = int(os.environ.get("SYMDIG_THREADS", cfg["runtime"].get("threads", 2)))
    cfg["runtime"]["num_workers"] = int(os.environ.get("SYMDIG_NUM_WORKERS", cfg["runtime"].get("num_workers", 2)))
    return cfg


def cmd_plan(cfg, args, exps):
    from symdig.planner import Planner
    q = Queue(args.results)
    p = saved_config_path(args.results)
    h = cfg_hash(cfg)
    if os.path.exists(p):
        old = yaml.safe_load(open(p))
        if cfg_hash(old) != h:
            print(f"[plan] NOTE: configuration differs from {p}; new jobs get new ids (old results are kept)")
    with open(p, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    specs = Planner(cfg, args.datasets, args.seeds, cfg_hash=h).plan(exps)
    added, by_exp = 0, {}
    for s in specs:
        jid, new = q.add(s)
        added += new
        by_exp.setdefault(s["exp"], []).append(jid)
    for e, ids in by_exp.items():
        q.record_index(e, ids)
    print(f"[plan] {len(specs)} job specs, {added} new. Per experiment: " +
          ", ".join(f"{e}: {len(set(v))}" for e, v in by_exp.items()))


def run_job(cfg, q, spec, log_to_stdout=True):
    from symdig.executors import EXECUTORS, resolved_cfg_for_record
    jid = spec["id"]
    lf = open(os.path.join(args_results(cfg), "logs", f"{jid}.log"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] [{jid[:8]} {spec['kind']} {spec['dataset']}] {msg}"
        lf.write(line + "\n")
        lf.flush()
        if log_to_stdout:
            print(line, flush=True)

    t0 = time.time()
    log(f"start: {json.dumps({k: v for k, v in spec.items() if k not in ('after', 'exps')})}")
    try:
        if spec["kind"] in ("gnn", "deploy", "tune", "evasion", "transfer", "tabular", "fecograph"):
            try:
                from symdig.trainer import setup_perf
                setup_perf(cfg["runtime"]["threads"])
            except ImportError:  # torch missing: only tabular jobs can run (sklearn fallback)
                pass
        res = EXECUTORS[spec["kind"]](cfg, spec, args_results(cfg), log)
        q.finish(spec, res, resolved_cfg_for_record(cfg, spec, args_results(cfg)), t0, ok=True)
        log(f"done in {(time.time() - t0) / 60:.1f} min")
        ok = True
    except Exception as ex:
        err = traceback.format_exc()
        log(f"FAILED: {ex}\n{err}")
        q.finish(spec, None, None, t0, ok=False, error=err)
        ok = False
    finally:
        lf.close()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return ok


def args_results(cfg):
    return cfg["paths"]["results"]


def cmd_worker(cfg, args):
    q = Queue(args.results)
    kinds = set(args.kinds) if args.kinds else None
    accept = (lambda s: s["kind"] in kinds) if kinds else None
    done, idle_since = 0, None
    while True:
        spec, _ = q.claim(accept)
        if spec is not None:
            idle_since = None
            run_job(cfg, q, spec)
            done += 1
            if args.max_jobs and done >= args.max_jobs:
                return
            continue
        pend = [q.spec(j) for j in q.list("pending")]
        pend = [s for s in pend if accept is None or accept(s)]
        if not pend:
            print("[worker] no pending job left, exiting")
            return
        if all(q.failed_deps(s) for s in pend):
            print(f"[worker] {len(pend)} pending job(s) depend on failed jobs; see `python run.py status`. Exiting.")
            return
        idle_since = idle_since or time.time()
        if not q.list("running") and time.time() - idle_since > 300:
            print("[worker] pending jobs wait for dependencies that nobody is running "
                  "(another worker with --kinds?). Exiting.")
            return
        time.sleep(15)


def cmd_status(cfg, args):
    q = Queue(args.results)
    counts = {s: len(q.list(s)) for s in ("pending", "running", "done", "failed")}
    print("queue:", counts)
    by = {}
    for s in ("pending", "running", "done", "failed"):
        for jid in q.list(s):
            sp = q.spec(jid)
            for e in sp.get("exps", [sp.get("exp")]):
                by.setdefault(e, {}).setdefault(s, 0)
                by[e][s] += 1
    for e in sorted(by):
        print(f"  {e:8s}", by[e])
    for jid in q.list("failed"):
        sp = q.spec(jid)
        last = (sp.get("error") or "").strip().splitlines()[-1:] or [""]
        print(f"  FAILED {jid} {sp['kind']} {sp['dataset']} {sp.get('model', '')} seed={sp.get('seed')}: {last[0]}")
        print(f"         log: {os.path.join(args.results, 'logs', jid + '.log')}")
    blocked = [j for j in q.list("pending") if q.failed_deps(q.spec(j))]
    if blocked:
        print(f"  {len(blocked)} pending job(s) depend on failed jobs (fix them, then `python run.py retry`)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "worker", "status", "run", "requeue", "retry", "smoke"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--kinds", nargs="+", default=None, help="worker: only these job kinds")
    ap.add_argument("--max-jobs", type=int, default=0)
    args = ap.parse_args()
    if args.results is None:
        args.results = "results_smoke" if args.cmd == "smoke" else "results"
    cfg = load_run_config(args)

    if args.cmd == "plan":
        bad = [x for x in args.rest if x not in EXPS]
        if bad or not args.rest:
            sys.exit(f"plan needs experiments from {EXPS}")
        cmd_plan(cfg, args, args.rest)
    elif args.cmd == "smoke":
        from symdig.planner import smoke_config
        cfg = smoke_config(cfg)
        cfg["paths"]["results"] = args.results
        args.datasets = args.datasets or ["ton"]
        exps = args.rest or ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
        if "E5" in exps and len(args.datasets) < 2:
            print("[smoke] E5 transfer needs two datasets; transfer jobs are skipped with one dataset")
        cmd_plan(cfg, args, exps)
        cmd_worker(cfg, args)
        cmd_status(cfg, args)
    elif args.cmd == "worker":
        cmd_worker(cfg, args)
    elif args.cmd == "status":
        cmd_status(cfg, args)
    elif args.cmd == "run":
        q = Queue(args.results)
        for jid in args.rest:
            spec = q.spec(jid)
            st = q.state_of(jid)
            if st in ("pending", "failed"):
                os.replace(q.p(st, jid), q.p("running", jid))
            run_job(cfg, q, spec)
    elif args.cmd == "requeue":
        print(f"moved {Queue(args.results).requeue(('running',))} job(s) back to pending")
    elif args.cmd == "retry":
        print(f"moved {Queue(args.results).requeue(('failed',))} job(s) back to pending")


if __name__ == "__main__":
    main()
