# PLACEHOLDER
# 4.3 CroCo-family adapters (DUSt3R, MASt3R, MonST3R): the asymmetric two-view network of the shared
#     naver/dust3r code base. Depth of view 1 is the z-coordinate of its pointmap; the single-image
#     reference is the image paired with itself (the monocular protocol of the DUSt3R paper).
#     (The v6.0 Easi3R attention probe on 10-frame clips was withdrawn after the first full run, D33.)

class CroCoModel(ReconModel):
    family, input_size, precision = "croco", (512, 384), "fp32"
    model_class = "dust3r"

    def import_code(self):
        if self.model_class == "mast3r":
            from mast3r.model import AsymmetricMASt3R as cls
        else:
            from dust3r.model import AsymmetricCroCo3DStereo as cls
        self._cls = cls

    def build(self):
        return self._cls.from_pretrained(self.checkpoint)

    def _tensor(self, img):
        x = torch.from_numpy(resize_rgb(img, self.input_size)).float().permute(2, 0, 1) / 255.0
        return (x - 0.5) / 0.5

    @torch.no_grad()
    def _forward(self, tens, names, pairs):
        W, H = self.input_size
        B = len(pairs)
        v1 = dict(img=torch.stack([tens[a] for a, b in pairs]).to(device),
                  true_shape=torch.tensor([[H, W]] * B, dtype=torch.int32),
                  instance=[names[a] for a, b in pairs], idx=[a for a, b in pairs])
        v2 = dict(img=torch.stack([tens[b] for a, b in pairs]).to(device),
                  true_shape=torch.tensor([[H, W]] * B, dtype=torch.int32),
                  instance=[names[b] for a, b in pairs], idx=[b for a, b in pairs])
        pred1, pred2 = self.model(v1, v2)
        return pred1, pred2

    def _depth_conf(self, pred, k, Hn, Wn):
        z = pred["pts3d"][k][..., 2].float().cpu().numpy()
        c = pred["conf"][k].float().cpu().numpy()
        return to_native(z, Hn, Wn), to_native(c, Hn, Wn)

    def pair(self, im0, im1):
        Hn, Wn = im0.shape[:2]
        p1, _ = self._forward([self._tensor(im0), self._tensor(im1)], ["0", "1"], [(0, 1)])
        d, c = self._depth_conf(p1, 0, Hn, Wn)
        return dict(depth=d, conf=c)

    def mono(self, im0):
        Hn, Wn = im0.shape[:2]
        p1, _ = self._forward([self._tensor(im0)], ["0"], [(0, 0)])
        d, c = self._depth_conf(p1, 0, Hn, Wn)
        return dict(depth=d, conf=c)

    def pairs_from_frame0(self, im0, others):
        # depth of frame 0 paired (as reference view) with each image in `others`
        Hn, Wn = im0.shape[:2]
        tens = [self._tensor(im0)] + [self._tensor(o) for o in others]
        names = ["0"] + [f"x{k}" for k in range(len(others))]
        out = []
        for k in range(1, len(tens)):
            p1, _ = self._forward(tens, names, [(0, k)])
            d, c = self._depth_conf(p1, 0, Hn, Wn)
            out.append(dict(depth=d, conf=c))
        return out


class DUSt3RModel(CroCoModel):
    name, training = "DUSt3R", "static"
    checkpoint = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

class MonST3RModel(CroCoModel):
    name, training = "MonST3R", "dynamic (DUSt3R fine-tuned on video)"
    checkpoint = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"

class MASt3RModel(CroCoModel):
    name, training, model_class = "MASt3R", "static", "mast3r"
    checkpoint = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

print("CroCo-family adapters ready: DUSt3R, MASt3R, MonST3R.")