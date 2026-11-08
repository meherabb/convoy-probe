# Models

Six frozen models with their public checkpoints. The notebook clones each official repository at a
pinned commit, imports the model code from source and downloads the checkpoint from the Hugging Face
Hub (notebook Sections 4.1–4.5); nothing is fine-tuned. No weights or third-party code are part of
this repository; each checkpoint and repository is used under its own licence.

| Model | Checkpoint (Hugging Face) | Code | Input, precision | Params | Peak GPU (T4) | Virtual KITTI in training |
|---|---|---|---|---:|---:|:---:|
| DUSt3R | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt` | `naver/dust3r` @ `4c24a6ebf04809f2cfe59915e51779c8984aaa40` | 512×384, fp32 | 571.2 M | 2.69 GiB | no |
| MASt3R | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | `naver/mast3r` @ `f5209afc300cec36239a7ac992263f36847bbba0` | 512×384, fp32 | 688.6 M | 3.15 GiB | yes |
| MonST3R | `Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt` | DUSt3R code base (same commit) | 512×384, fp32 | 571.2 M | 2.69 GiB | no |
| VGGT-1B | `facebook/VGGT-1B` | `facebookresearch/vggt` @ `a288dd0f14786c93483e45524328726ab7b1b4ce` | 504×378, fp16 | 1190.6 M | 5.26 GiB | yes |
| π³ | `yyfz233/Pi3` | `yyfz/Pi3` @ `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` | 560×420, fp16 | 958.7 M | 3.75 GiB | no |
| MapAnything | `facebook/map-anything` | `facebookresearch/map-anything` @ `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | 504×378, fp32 | 1228.5 M | 5.84 GiB | no |

Parameters, peak memory and load status of the run: `results/v6.6/logs/model_status.csv` and
`results/v6.6/tables/T1a_models.csv`; the commits actually checked out: `results/v6.6/manifest/repos.json`.

**Protocol.** Every input is an exact rescaling of a native 4:3 frame (640×480; 500×375 for Virtual
KITTI 2), so model pixels map back to native pixels by a scale. The depth of the first view is read
(for the CroCo-family models the z-coordinate of its pointmap). A model's single-image reference is
its own prediction from one image: for the pairwise models the image paired with itself, as in the
DUSt3R paper; for the multi-view models the image given once. As a check (notebook Section 6.7d), the
multi-view models also received the image twice; their single-image results agree under both
protocols (`results/v6.6/results/single_image_protocol_check.csv`). The multi-view models also
receive K = 2, 4 and 8 frames jointly on the straight and curved camera paths of E4.

**Other networks used by the notebook.** Optical flow: RAFT-large with the torchvision weights
`C_T_SKHT_V2` (OpenCV DIS is the fallback if the weights cannot be downloaded; the reported run used
RAFT everywhere, `results/v6.6/manifest/flow.json`). People in the Bonn sequences are masked with
torchvision's Mask R-CNN ResNet-50 FPN v2 (COCO weights).

**Determinism.** A repeated forward pass of the same inputs gives bit-identical outputs in fp32 and
with AMP (`results/v6.6/manifest/env_log.json`, `determinism_check`).
