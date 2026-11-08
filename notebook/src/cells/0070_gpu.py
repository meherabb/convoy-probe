# PLACEHOLDER
# 0.3.3 GPU requirement and numerical precision policy.
import numpy as np
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. This notebook requires one GPU (T4-class). "
        "Kaggle: Settings -> Accelerator -> GPU T4. Colab: Runtime -> Change runtime type -> GPU.")
torch.cuda.set_device(0)
device = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_CAPABILITY = torch.cuda.get_device_capability(0)
GPU_TOTAL_GIB = torch.cuda.get_device_properties(0).total_memory / 2**30

# Precision policy. bfloat16 needs compute capability >= 8.0 (Ampere). On a T4 (7.5)
# bf16 is either emulated (slow) or, for code that forces FlashAttention when it sees
# bf16 inputs (pi3), fails with "No available kernel" -- the failure seen in the
# previous version of this notebook. fp16 is the documented fallback of VGGT and pi3.
AMP_DTYPE = torch.bfloat16 if GPU_CAPABILITY[0] >= 8 else torch.float16

print(f"GPU            : {GPU_NAME}  (compute capability {GPU_CAPABILITY[0]}.{GPU_CAPABILITY[1]}, "
      f"{GPU_TOTAL_GIB:.1f} GiB)")
print(f"CUDA / torch   : {torch.version.cuda} / {torch.__version__}")
print(f"Mixed precision: {AMP_DTYPE} (bf16 only when compute capability >= 8.0)")
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs visible; only cuda:0 is used.")
if "T4" not in GPU_NAME:
    print(f"Note: GPU is '{GPU_NAME}', not a T4. Sample sizes are unchanged; runtimes will differ.")