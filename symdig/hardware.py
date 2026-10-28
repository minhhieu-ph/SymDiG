"""GPU memory / power monitoring (pynvml is optional)."""
import threading
import time

import torch


class PowerMonitor:
    """Samples GPU board power (W) in a background thread."""

    def __init__(self, interval=0.2, device_index=0):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self.nvml = None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        if self.handle is not None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self.handle is not None:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {"power_mean_w": None, "power_peak_w": None}
        s = sorted(self.samples)
        return {"power_mean_w": sum(s) / len(s), "power_peak_w": s[-1],
                "power_p95_w": s[int(0.95 * (len(s) - 1))]}


def gpu_info():
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    p = torch.cuda.get_device_properties(0)
    return {"device": p.name, "total_vram_gb": p.total_memory / 2**30}


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return None
