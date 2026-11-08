# PLACEHOLDER
# 0.3.1 Environment detection and canonical paths (single source of truth).
import os, sys, platform, json, time, subprocess, importlib, importlib.util, shutil, gc, math
import warnings, hashlib, glob, re, io, contextlib, itertools, textwrap

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # keep the outputs readable
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)        # deprecation notices from model code
warnings.filterwarnings("ignore", message=".*xFormers.*")
warnings.filterwarnings("ignore", message="Mean of empty slice")                  # nan-aware summaries of empty groups
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="An input array is constant")          # Spearman on a constant prior weight


def _detect_env():
    # Kaggle is checked first: recent Kaggle images also ship the google.colab package, and the
    # previous version of this notebook took a Kaggle session for Colab and wrote its outputs to
    # /content, which Kaggle does not keep (correction D19).
    if os.path.isdir("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


ENV_NAME = _detect_env()
_ROOT_OVERRIDE = os.environ.get("CONVOY_ROOT")
if _ROOT_OVERRIDE:
    ROOT_WORK = os.path.abspath(_ROOT_OVERRIDE)
elif ENV_NAME == "kaggle":
    ROOT_WORK = "/kaggle/working"
elif ENV_NAME == "colab":
    ROOT_WORK = "/content"
else:
    ROOT_WORK = os.path.abspath(".")

OUTPUT_DIR   = os.path.join(ROOT_WORK, "outputs")              # results, figures, tables, logs, manifests
CKPT_DIR     = os.path.join(ROOT_WORK, "checkpoints")          # one archive per finished stage (resume points)
ZIP_PATH     = os.path.join(ROOT_WORK, "convoy_checkpoints.zip")    # resume archive: every finished stage, nothing else
PAPER_ZIP    = os.path.join(ROOT_WORK, "convoy_paper_outputs.zip")  # figures, tables, results, manifests for the paper
LOG_DIR      = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR   = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR    = os.path.join(OUTPUT_DIR, "tables")               # formatted paper tables (csv/json/tex)
RESULTS_DIR  = os.path.join(OUTPUT_DIR, "results")              # unformatted numeric results of every analysis
MANIFEST_DIR = os.path.join(OUTPUT_DIR, "manifest")
CACHE_DIR    = os.path.join(OUTPUT_DIR, "cache")               # compact per-model measurements

if _ROOT_OVERRIDE:
    SCRATCH_DIR = os.path.join(ROOT_WORK, "scratch")
elif ENV_NAME in ("kaggle", "colab"):
    SCRATCH_DIR = "/tmp/convoy_scratch"                         # large working data; archived per stage in checkpoints/
else:
    SCRATCH_DIR = os.path.join(ROOT_WORK, "convoy_scratch")
DATA_DIR    = os.path.join(SCRATCH_DIR, "data")                 # downloaded datasets
RENDER_DIR  = os.path.join(SCRATCH_DIR, "renders")              # CONVOY-3D renders and CONVOY-REAL composites
FLOW_DIR    = os.path.join(SCRATCH_DIR, "flow")                 # optical-flow cache
VK_DIR      = os.path.join(SCRATCH_DIR, "vkitti")               # Virtual KITTI 2 frames used by this run
REPO_DIR    = os.path.join(SCRATCH_DIR, "repos")                # cloned model code

for _p in [OUTPUT_DIR, CKPT_DIR, LOG_DIR, FIGURE_DIR, TABLE_DIR, RESULTS_DIR, MANIFEST_DIR, CACHE_DIR,
           SCRATCH_DIR, DATA_DIR, RENDER_DIR, FLOW_DIR, VK_DIR, REPO_DIR]:
    os.makedirs(_p, exist_ok=True)
# Figures, tables and analysis results are rebuilt by every run from the stage outputs, so files left
# by an earlier run in the same folder (e.g. a figure this run does not draw) are removed first.
_n_old = 0
for _p in [FIGURE_DIR, TABLE_DIR, RESULTS_DIR]:
    for _fn in os.listdir(_p):
        if os.path.isfile(os.path.join(_p, _fn)):
            os.remove(os.path.join(_p, _fn)); _n_old += 1
if _n_old:
    print(f"Removed {_n_old} figure/table/result files of an earlier run in this folder (they are rebuilt below).")

for _k in ["ENV_NAME", "ROOT_WORK", "OUTPUT_DIR", "CKPT_DIR", "ZIP_PATH", "PAPER_ZIP", "LOG_DIR", "FIGURE_DIR", "TABLE_DIR",
           "RESULTS_DIR", "MANIFEST_DIR", "CACHE_DIR", "SCRATCH_DIR", "DATA_DIR", "RENDER_DIR", "FLOW_DIR",
           "VK_DIR", "REPO_DIR"]:
    print(f"{_k:<13}= {globals()[_k]}")
_du = shutil.disk_usage(SCRATCH_DIR)
print(f"\nFree disk at scratch location: {_du.free / 2**30:.1f} GiB")
if ENV_NAME == "colab":
    print("Colab: files in /content disappear when the runtime ends. Download convoy_checkpoints.zip at the end "
          "(or copy checkpoints/ to Google Drive) to resume later.")
# ---- memory (D32): allocator settings, explicit release, and a report at every milestone ----
# Large numpy/torch buffers are mapped from the OS and handed back when freed (instead of being kept
# in fragmented heaps), the number of allocator arenas is capped for the rendering threads, and
# release_memory() returns everything that is free to the OS. mem_report() prints this process's
# resident memory, the memory still available to it (container limit, page cache excluded) and the
# GPU memory in use; MEMORY_LOG is written to outputs/manifest/memory_log.csv.
import ctypes
import numpy as np
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.mallopt(-3, 8 << 20)                 # M_MMAP_THRESHOLD: blocks >= 8 MiB are mmapped
    _LIBC.mallopt(-1, 64 << 20)                # M_TRIM_THRESHOLD
    _LIBC.mallopt(-8, 4)                       # M_ARENA_MAX
except (OSError, AttributeError):
    _LIBC = None
MEMORY_LOG = []


def _read_first(paths):
    for p in paths:
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def ram_status():
    # GiB: process resident set, memory available to this process, and the limit it lives under
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) / 2**20
    except OSError:
        pass
    rss = np.nan
    try:
        with open("/proc/self/status") as f:
            rss = next(int(l.split()[1]) / 2**20 for l in f if l.startswith("VmRSS"))
    except (OSError, StopIteration):
        pass
    total, avail = info.get("MemTotal", np.nan), info.get("MemAvailable", np.nan)
    lim = _read_first(["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"])
    use = _read_first(["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"])
    stat = _read_first(["/sys/fs/cgroup/memory.stat", "/sys/fs/cgroup/memory/memory.stat"]) or ""
    if lim and lim != "max" and use and int(lim) < (1 << 50):
        # file-backed page cache (downloaded weights, datasets) is counted in the cgroup's usage but is
        # reclaimed on demand; only shared memory (tmpfs) is not (D32, corrected in v6.2)
        kv = {l.split()[0]: int(l.split()[1]) for l in stat.splitlines() if len(l.split()) == 2 and l.split()[1].isdigit()}
        cache = kv.get("file", kv.get("total_cache", kv.get("inactive_file", kv.get("total_inactive_file", 0))))
        shmem = kv.get("shmem", kv.get("total_shmem", 0))
        reclaim = max(cache - shmem, 0)
        total = int(lim) / 2**30
        avail = total - (int(use) - reclaim) / 2**30
    return dict(rss=rss, available=avail, total=total)


def release_memory():
    gc.collect(); gc.collect()
    if "torch" in sys.modules:
        _t = sys.modules["torch"]
        try:
            if _t.cuda.is_available() and _t.cuda.is_initialized():
                _t.cuda.synchronize(); _t.cuda.empty_cache(); _t.cuda.ipc_collect()
            if hasattr(_t._C, "_host_emptyCache"):
                _t._C._host_emptyCache()               # pinned host memory cached by torch
        except Exception:
            pass
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def mem_report(tag, quiet=False):
    r = ram_status()
    gpu = (np.nan, np.nan)
    if "torch" in sys.modules:
        _t = sys.modules["torch"]
        try:
            if _t.cuda.is_available() and _t.cuda.is_initialized():
                gpu = (_t.cuda.memory_allocated() / 2**30, _t.cuda.memory_reserved() / 2**30)
        except Exception:
            pass
    MEMORY_LOG.append(dict(time_utc=time.strftime("%H:%M:%S", time.gmtime()), milestone=tag, process_rss_gib=r["rss"],
                           available_gib=r["available"], total_gib=r["total"], gpu_allocated_gib=gpu[0],
                           gpu_reserved_gib=gpu[1]))
    if not quiet:
        g = f" | GPU {gpu[0]:.1f} GiB in use ({gpu[1]:.1f} reserved)" if np.isfinite(gpu[0]) else ""
        print(f"[memory] {tag}: this process {r['rss']:.1f} GiB | available {r['available']:.1f} of {r['total']:.1f} GiB{g}")
    return r


release_memory(); _ = mem_report("start")
