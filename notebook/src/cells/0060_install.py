# PLACEHOLDER
# 0.3.2 Minimal dependency installation under a constraints file.
#       Freezes the preinstalled numerical stack so no install below can replace it.
from importlib import metadata as _md

_FREEZE_DISTS = ["torch", "torchvision", "torchaudio", "numpy", "opencv-python",
                 "opencv-python-headless", "opencv-contrib-python", "pillow", "scipy",
                 "pandas", "matplotlib"]
_constraint_lines = []
for _d in _FREEZE_DISTS:
    try:
        _constraint_lines.append(f"{_d}=={_md.version(_d)}")
    except _md.PackageNotFoundError:
        pass
CONSTRAINTS_FILE = os.path.join(SCRATCH_DIR, "pip_constraints.txt")
with open(CONSTRAINTS_FILE, "w") as _f:
    _f.write("\n".join(_constraint_lines) + "\n")

# (import name, pip requirement, needed by)
REQUIRED_PACKAGES = [
    ("numpy", "numpy", "everything"),
    ("scipy", "scipy", "statistics, root finding"),
    ("pandas", "pandas", "result tables"),
    ("pyarrow", "pyarrow", "Argoverse 2 tables and LiDAR sweeps (feather)"),
    ("matplotlib", "matplotlib", "figures"),
    ("PIL", "pillow", "image I/O"),
    ("cv2", "opencv-python-headless", "DIS optical flow, PnP, image ops"),
    ("torch", "torch", "models"),
    ("torchvision", "torchvision", "RAFT optical flow"),
    ("huggingface_hub", "huggingface_hub", "checkpoint download"),
    ("safetensors", "safetensors", "checkpoint loading"),
    ("einops", "einops", "DUSt3R family, VGGT"),
    ("roma", "roma", "DUSt3R family"),
    ("omegaconf", "omegaconf", "MapAnything"),
    ("hydra", "hydra-core", "MapAnything"),
    ("box", "python-box", "MapAnything"),
    ("orjson", "orjson", "MapAnything"),
    ("natsort", "natsort", "MapAnything"),
    ("jaxtyping", "jaxtyping", "MapAnything (uniception)"),
    ("timm", "timm", "MapAnything (uniception)"),
    # installed WITHOUT dependencies: its requirement list includes torchaudio and other
    # packages that the model code never imports; the ones it does import are listed above
    ("uniception", "uniception==0.1.7 --no-deps", "MapAnything (encoders and heads)"),
]

INSTALL_LOG = []
for _mod, _req, _why in REQUIRED_PACKAGES:
    _status = "preinstalled"
    if importlib.util.find_spec(_mod) is None:
        _cmd = [sys.executable, "-m", "pip", "install", "-q", "-c", CONSTRAINTS_FILE] + _req.split()
        _rc = subprocess.run(_cmd, capture_output=True, text=True, timeout=1800)
        importlib.invalidate_caches()
        if _rc.returncode == 0 and importlib.util.find_spec(_mod) is not None:
            _status = "installed now"
        else:
            _status = "INSTALL FAILED"
            print(f"[install] {_req} failed:\n{_rc.stderr[-1500:]}")
    _dist = _req.split()[0].split("==")[0]
    try:
        _ver = _md.version(_dist)
    except _md.PackageNotFoundError:
        _ver = "-"
    INSTALL_LOG.append(dict(package=_dist, import_name=_mod, version=_ver, status=_status, needed_by=_why))

print(f"Constraints file ({len(_constraint_lines)} frozen distributions): {CONSTRAINTS_FILE}")
for _l in _constraint_lines:
    print("   ", _l)
print()
print(f"{'package':<26}{'version':<14}{'status':<16}needed by")
for _r in INSTALL_LOG:
    print(f"{_r['package']:<26}{_r['version']:<14}{_r['status']:<16}{_r['needed_by']}")
_n_fail = sum(r["status"] == "INSTALL FAILED" for r in INSTALL_LOG)
print(f"\n{len(INSTALL_LOG) - _n_fail}/{len(INSTALL_LOG)} required packages available.")
with open(os.path.join(LOG_DIR, "install_log.json"), "w") as _f:
    json.dump(INSTALL_LOG, _f, indent=2)