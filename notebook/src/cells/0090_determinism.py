# PLACEHOLDER
# 0.3.5 Determinism self-check: identical inputs must give identical outputs.
_rng_a = np.random.default_rng(seed_for("check")).standard_normal(5)
_rng_b = np.random.default_rng(seed_for("check")).standard_normal(5)
assert np.array_equal(_rng_a, _rng_b), "NumPy generator is not reproducible"

_g = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
_x = torch.randn(2, 3, 64, 64, generator=_g).to(device)
_conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
with torch.no_grad():
    _y1 = _conv(_x)
    _y2 = _conv(_x)
    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE):
        _y3 = _conv(_x).float()
        _y4 = _conv(_x).float()
_det_fp32 = float((_y1 - _y2).abs().max())
_det_amp = float((_y3 - _y4).abs().max())
print(f"NumPy generator reproducible           : True")
print(f"Repeated fp32 convolution, max |diff|  : {_det_fp32:.2e}")
print(f"Repeated {str(AMP_DTYPE).split('.')[-1]} convolution, max |diff|: {_det_amp:.2e}")
assert _det_fp32 == 0.0 and _det_amp == 0.0, "repeated forward passes differ"
ENV_LOG["determinism_check"] = {"fp32_max_abs_diff": _det_fp32, "amp_max_abs_diff": _det_amp}
del _x, _conv, _y1, _y2, _y3, _y4