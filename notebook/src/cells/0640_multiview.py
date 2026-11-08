# PLACEHOLDER
# 4.4 Multi-view adapters: VGGT-1B, pi3, MapAnything. Each takes N >= 1 frames jointly;
#     the depth of frame 0 is returned at native resolution.

class MultiViewModel(ReconModel):
    family = "multiview"

    def pair(self, im0, im1):
        return self.multi([im0, im1])

    def mono(self, im0):
        return self.multi([im0])

class VGGTModel(MultiViewModel):
    name, training, checkpoint = "VGGT-1B", "multi-dataset mixture", "facebook/VGGT-1B"
    input_size = (504, 378)

    def __init__(self):
        super().__init__()
        self.precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(AMP_DTYPE, "fp32")

    def import_code(self):
        from vggt.models.vggt import VGGT
        self._cls = VGGT

    def build(self):
        m = self._cls.from_pretrained(self.checkpoint)
        m.track_head = None                       # tracking head is not used
        return m

    @torch.no_grad()
    def multi(self, ims):
        Hn, Wn = ims[0].shape[:2]
        x = torch.stack([torch.from_numpy(resize_rgb(im, self.input_size, _PIL.BICUBIC)).float().permute(2, 0, 1) / 255.0
                         for im in ims])[None].to(device)
        with self.autocast():
            pred = self.model(x)
        d = pred["depth"][0, 0, :, :, 0].float().cpu().numpy()
        c = pred["depth_conf"][0, 0].float().cpu().numpy()
        return dict(depth=to_native(d, Hn, Wn), conf=to_native(c, Hn, Wn))

class Pi3Model(MultiViewModel):
    name, training, checkpoint = "pi3", "multi-dataset mixture", "yyfz233/Pi3"
    input_size = (560, 420)

    def __init__(self):
        super().__init__()
        self.precision = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(AMP_DTYPE, "fp32")

    def import_code(self):
        from pi3.models.pi3 import Pi3
        self._cls = Pi3

    def build(self):
        return self._cls.from_pretrained(self.checkpoint)

    @torch.no_grad()
    def multi(self, ims):
        Hn, Wn = ims[0].shape[:2]
        x = torch.stack([torch.from_numpy(resize_rgb(im, self.input_size)).float().permute(2, 0, 1) / 255.0
                         for im in ims])[None].to(device)
        with self.autocast():
            res = self.model(x)
        d = res["local_points"][0, 0, :, :, 2].float().cpu().numpy()
        c = torch.sigmoid(res["conf"][0, 0, :, :, 0].float()).cpu().numpy()
        return dict(depth=to_native(d, Hn, Wn), conf=to_native(c, Hn, Wn))

class MapAnythingModel(MultiViewModel):
    name, training, checkpoint = "MapAnything", "multi-dataset mixture", "facebook/map-anything"
    input_size, precision = (504, 378), "fp32"

    def import_code(self):
        from mapanything.models import MapAnything
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
        self._cls, self._norms = MapAnything, IMAGE_NORMALIZATION_DICT

    def build(self):
        return self._cls.from_pretrained(self.checkpoint)

    @torch.no_grad()
    def multi(self, ims):
        Hn, Wn = ims[0].shape[:2]
        W, H = self.input_size
        norm_type = self.model.encoder.data_norm_type
        nz = self._norms[norm_type]
        mean = torch.as_tensor(nz.mean).view(3, 1, 1); std = torch.as_tensor(nz.std).view(3, 1, 1)
        views = []
        for i, im in enumerate(ims):
            x = torch.from_numpy(resize_rgb(im, self.input_size)).float().permute(2, 0, 1) / 255.0
            views.append(dict(img=((x - mean) / std)[None], data_norm_type=[norm_type],
                              true_shape=np.int32([[H, W]]), idx=i, instance=str(i)))
        preds = self.model.infer(views, memory_efficient_inference=True, use_amp=False,
                                 apply_mask=False, mask_edges=False)
        d = preds[0]["depth_z"][0, :, :, 0].float().cpu().numpy()
        c = preds[0]["conf"][0].float().cpu().numpy()
        return dict(depth=to_native(d, Hn, Wn), conf=to_native(c, Hn, Wn))

print("Multi-view adapters ready: VGGT-1B, pi3, MapAnything.")