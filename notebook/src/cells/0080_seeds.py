# PLACEHOLDER
# 0.3.4 Seeds, deterministic settings and the environment record.
import random
import pandas as pd
import scipy
import matplotlib
import PIL
import cv2

GLOBAL_SEED = 20270918
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Full torch.use_deterministic_algorithms(True) is NOT enabled: several attention and
# upsampling kernels used by the evaluated models have no deterministic CUDA
# implementation and would raise. Inference-only forward passes are reproducible to
# within fp16 rounding; Section 0.3.5 measures this directly.
cv2.setNumThreads(max(1, (os.cpu_count() or 2)))

def seed_for(*keys):
    # Stable 32-bit seed derived from GLOBAL_SEED and any hashable keys.
    h = hashlib.sha256(("|".join(map(str, (GLOBAL_SEED,) + keys))).encode()).hexdigest()
    return int(h[:8], 16)

ENV_LOG = {
    "env_name": ENV_NAME, "python": sys.version.split()[0], "platform": platform.platform(),
    "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
    "matplotlib": matplotlib.__version__, "pillow": PIL.__version__, "opencv": cv2.__version__,
    "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": GPU_NAME,
    "gpu_capability": f"{GPU_CAPABILITY[0]}.{GPU_CAPABILITY[1]}", "amp_dtype": str(AMP_DTYPE),
    "cpu_count": os.cpu_count(), "global_seed": GLOBAL_SEED,
    "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(os.path.join(LOG_DIR, "env_log.json"), "w") as _f:
    json.dump(ENV_LOG, _f, indent=2)
for _k, _v in ENV_LOG.items():
    print(f"{_k:<16}{_v}")
RUN_T0 = time.time()
STAGE_TIMES = {}

@contextlib.contextmanager
def stage(name):
    # Wall-clock timer for each expensive stage; reported in Section 13.
    t0 = time.time()
    try:
        yield
    finally:
        STAGE_TIMES[name] = STAGE_TIMES.get(name, 0.0) + time.time() - t0