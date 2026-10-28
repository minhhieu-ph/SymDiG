"""torch DataLoader around the numpy sampler (sampling and feature gathering run in workers)."""
import os

import numpy as np
import torch

from .sampler import build_batch, twin_coverage


class SubgraphBatches(torch.utils.data.Dataset):
    """Item i = the computation subgraph of the i-th chunk of targets."""

    def __init__(self, graph, store, labels, targets, batch_size, mode, cfg_s, shuffle, seed,
                 epoch=0, max_batches=None):
        self.graph, self.store, self.labels = graph, store, labels
        self.mode, self.cfg_s = dict(mode), cfg_s
        self.seed, self.epoch = int(seed), int(epoch)
        t = np.asarray(targets, np.int64)
        if shuffle:
            t = np.random.default_rng([self.seed, self.epoch, 7]).permutation(t)
        else:
            t = np.sort(t)
        self.chunks = [t[i:i + batch_size] for i in range(0, len(t), batch_size)]
        if max_batches:
            self.chunks = self.chunks[:max_batches]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        b = build_batch(self.graph, self.chunks[i], self.mode, self.cfg_s, rng)
        n = len(b["nodes"])
        ego = np.zeros(n, np.float32)
        ego[b["ego"]] = 1.0
        x_node = np.stack([np.ones(n, np.float32), ego], 1)
        pt = b["targets"][b["present"]]
        out = {
            "num_nodes": n,
            "x_node": torch.from_numpy(x_node),
            "ei_f": torch.from_numpy(np.ascontiguousarray(b["ei_f"])),
            "ei_r": torch.from_numpy(np.ascontiguousarray(b["ei_r"])),
            "xf": torch.from_numpy(self.store.gather(b["ef_ids"])),
            "shared": b["shared"],
            "tpos": torch.from_numpy(b["tpos"]),
            "targets": torch.from_numpy(b["targets"]),
            "present": torch.from_numpy(b["present"]),
            "y": torch.from_numpy(self.labels[pt].astype(np.int64)),
            "n_edges": int(len(b["ef_ids"]) + (0 if b["shared"] else len(b["er_ids"]))),
            "twin": twin_coverage(b),
        }
        if not b["shared"]:
            out["xr"] = torch.from_numpy(self.store.gather(b["er_ids"]))
        return out


def make_loader(dataset, num_workers):
    if os.name == "nt":
        num_workers = 0
    kw = {}
    if num_workers > 0:
        # fork: workers share the memory-mapped arrays instead of pickling them
        # (Python >= 3.14 defaults to forkserver on Linux, which would copy the arrays)
        kw = {"prefetch_factor": 4, "persistent_workers": False, "multiprocessing_context": "fork"}
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=num_workers,
                                       pin_memory=torch.cuda.is_available(), **kw)


def to_device(b, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
