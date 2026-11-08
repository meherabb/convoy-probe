# PLACEHOLDER
# 4.2 Common adapter interface. Every adapter takes native 4:3 uint8 RGB images (640x480, or 500x375
#     for Virtual KITTI 2) and returns depth (and confidence) maps resampled to the native pixel grid.
from PIL import Image as _PIL

def resize_rgb(img, size, method=_PIL.LANCZOS):
    return np.array(_PIL.fromarray(img).resize(size, method))            # writable copy (torch.from_numpy)

def to_native(arr, H, W):
    return cv2.resize(np.asarray(arr, np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

class ReconModel:
    name, family, training, checkpoint = "", "", "", ""
    input_size, precision = (0, 0), "fp32"

    def __init__(self):
        self.model = None
        self.status, self.note, self.n_params, self.load_seconds = "NOT LOADED", "", None, None

    # -- lifecycle ------------------------------------------------------------------
    def import_code(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def load(self):
        t0 = time.time()
        self.import_code()
        m = self.build().eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m
        self.n_params = sum(p.numel() for p in m.parameters())
        self.load_seconds = time.time() - t0
        return self

    def to_gpu(self):
        self.model.to(device)
        return self

    def unload(self):
        # drop every reference to the weights and hand the memory back (GPU and host); the model's
        # results are already archived when this is called
        self.model = None
        release_memory()

    def autocast(self):
        if self.precision == "fp32":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=AMP_DTYPE)

    # -- inference (implemented per family) --------------------------------------------
    def pair(self, im0, im1):
        # depth/conf of view 0 given views (0, 1)
        raise NotImplementedError

    def mono(self, im0):
        raise NotImplementedError

    def multi(self, ims):
        raise NotImplementedError

print("Adapter base ready.")