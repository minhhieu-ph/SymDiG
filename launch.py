"""Start one worker process per GPU (or several per GPU) on this machine / pod.

  python launch.py                          # all visible GPUs, 1 worker each, CPUs split evenly
  python launch.py --gpus 0 1 2 3 --workers-per-gpu 1 --cpus 11
  python launch.py --kinds prepare coverage sizes survival --gpus none --workers 4   # CPU-only jobs

Several pods (Kubeflow) can run launch.py at the same time against the same --results and
cache directories on a shared volume: jobs are claimed by atomic rename, so each job runs once.
Workers exit when the queue is empty. Logs: results/logs/<job>.log and results/workers/*.log.
"""
import argparse
import os
import subprocess
import sys
import time


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", nargs="+", default=None, help="GPU indices, or 'none' for CPU-only workers")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--workers", type=int, default=None, help="number of CPU-only workers (with --gpus none)")
    ap.add_argument("--cpus", type=int, default=os.cpu_count(), help="CPU cores to share among the workers")
    ap.add_argument("--results", default="results")
    ap.add_argument("--kinds", nargs="+", default=None)
    args = ap.parse_args()

    if args.gpus == ["none"]:
        slots = [None] * (args.workers or 1)
    else:
        gpus = args.gpus or detect_gpus()
        if not gpus:
            sys.exit("no GPU found; use --gpus none --workers N for CPU-only jobs")
        slots = [g for g in gpus for _ in range(args.workers_per_gpu)]
    n = len(slots)
    threads = max(1, args.cpus // n)
    loader_workers = max(1, threads - 1)
    os.makedirs(os.path.join(args.results, "workers"), exist_ok=True)
    procs = []
    for i, g in enumerate(slots):
        env = dict(os.environ, SYMDIG_THREADS=str(threads), SYMDIG_NUM_WORKERS=str(loader_workers),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        env["CUDA_VISIBLE_DEVICES"] = "" if g is None else str(g)
        cmd = [sys.executable, "run.py", "worker", "--results", args.results]
        if args.kinds:
            cmd += ["--kinds", *args.kinds]
        log = open(os.path.join(args.results, "workers", f"worker{i}_gpu{g}.log"), "a")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"worker {i}: GPU {g}, {threads} threads, {loader_workers} sampling workers")
        time.sleep(2)  # stagger start-up (first jobs read the same files)
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(30)
    except KeyboardInterrupt:
        for p, _ in procs:
            p.terminate()
    for p, log in procs:
        log.close()
    print("all workers finished; `python run.py status` for a summary")


if __name__ == "__main__":
    main()
