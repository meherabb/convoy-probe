# Data

The notebook generates the synthetic testbeds itself and downloads the public datasets from their
official servers; nothing needs to be prepared by hand. Every real sequence passes a ground-truth
self-consistency check (A3: depth warped with the given poses against the measured depth) before
it is used; the results of the check are in `results/v6.6/tables/T1b_data.csv` and in
`results/v6.6/manifest/*.json`.

## Synthetic testbeds (generated)

| Set | Notebook | Items | Movers | Used for |
|---|---|---:|---:|---|
| CONVOY-3D main (incl. knockout views) | 3.1–3.2 | 240 scenes | 715 | P1, A1, E2 masks, E3, robustness |
| CONVOY-3D multi-view (straight and curved twins) | 3.2 | 96 clips | 254 | P4 / E4 |
| E5 contact ablation, 5 conditions | 3.1.7 | 360 images (72 scenes) | 208 | H5–H8 |
| Baseline sweep, ×0.5 / ×1 / ×2 / ×3 | 3.2b | 256 images (64 scenes) | 192 | H12 |
| Dose-response, no wall / 15 / 35 / 70 cm | 3.2b | 256 images (64 scenes) | 190 | H13 |
| LEGACY-FLAT (flat textured backgrounds) | 3.1.6 | 24 | – | negative control of the gate |

CONVOY-3D is ray cast analytically (boxes on a street plane, procedural mip-mapped textures), so
depth, correspondences and the motion of every mover are exact. The main set was drawn from a
design deck of 240 × 3 movers; a mover that cannot be placed in its scene is offered again to the
next one (notebook cell 3.1.4), so scenes hold one to four movers and 715 of the 720 were placed. Movers have
`s ∈ [−0.6, 0.85]`, except 8% at `s ∈ [1.1, 1.4]`, which have no static explanation. The mover
tables (`results/v6.6/manifest/convoy3d_*_movers.csv`) list the design and the exact ground truth
of every mover; `tests/test_probe.py` checks that their regimes and sample flags follow from the
registered rules.

## Real data (downloaded by the notebook)

| Dataset | What is used | Role | Licence |
|---|---|---|---|
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | fr3 `structure_texture_far`, `long_office_household` | real backgrounds of CONVOY-REAL and E5-REAL (inserted crates with exact depth) | CC BY 4.0 |
| TUM RGB-D | fr3 `walking_xyz`, `walking_halfsphere` | prevalence of the motion regimes of walking people | CC BY 4.0 |
| [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2) | Scenes 01, 02, 06, 18, 20; `clone`, Camera_0; centred 500×375 crops | photo-realistic driving with exact depth, poses and instances | CC BY-NC-SA 3.0, non-commercial |
| [Argoverse 2 Sensor](https://www.argoverse.org/about.html) | 64 validation logs, `ring_front_center`, LiDAR, labelled boxes | real driving (exploratory), the twin test and H14 | CC BY-NC-SA 4.0 and the Argoverse Terms of Use |
| [Bonn RGB-D Dynamic](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) | `balloon`, `balloon2`, `moving_nonobstructing_box`, `moving_nonobstructing_box2` | real objects moved by people (exploratory; not analysed in the paper) | no licence stated; the authors ask to be cited |
| KITTI tracking | optional, only if attached as a private dataset | not part of the reported run | – |

Download sources and the read paths are set in notebook cell 0.4 and can be redirected with
environment variables (`CONVOY_AV2_BASE`, `CONVOY_VKITTI_BASE`, `CONVOY_BONN_BASE`,
`CONVOY_KITTI_ROOT`). Both driving datasets are read selectively: Virtual KITTI 2 with HTTP range
requests into its archives where the server supports them (streamed otherwise), Argoverse 2 file by
file from its public bucket, so only a small part of each dataset is transferred.

Please cite the datasets you use (references in the paper):
Sturm et al., IROS 2012 (TUM RGB-D); Gaidon et al., CVPR 2016 and Cabon et al., arXiv 2020
(Virtual KITTI and Virtual KITTI 2); Wilson et al., NeurIPS Datasets and Benchmarks 2021
(Argoverse 2); Palazzolo et al., IROS 2019 (Bonn RGB-D Dynamic).

## Licence of the shipped outputs

The code of this repository is MIT licensed. The run outputs in `results/v6.6/` are derived from the
datasets above and remain subject to their terms. Results derived from Virtual KITTI 2 and Argoverse 2
(for example `manifest/vkitti_movers.csv` and the twin-test results) and the figures that show their
frames (`figures/fig2_testbeds_checks.png`, `fig7_driving.png`, `fig9_qualitative.png`, `figS2_gallery.png`,
`figS4_av2_contact_twin.png`) are shared for non-commercial research under CC BY-NC-SA; results and
figures built on TUM RGB-D frames (CONVOY-REAL, E5-REAL, `fig5_real_contact.png`) under CC BY 4.0 with
attribution; `fig14_bonn.png` shows frames of the Bonn RGB-D Dynamic dataset. No model weights and no
third-party code are redistributed (see [MODELS.md](MODELS.md)).
