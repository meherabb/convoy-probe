# Reproducing the results

There are two levels. On a CPU, in a few minutes, the numbers of the paper can be recomputed from
the archived outputs of the reported run and the model-independent code can be checked against the
notebook. On a GPU, the notebook regenerates the testbeds, runs the six models and rebuilds all
outputs.

## 1. On a CPU: check the reported run

```bash
pip install -e ".[dev]"
python scripts/verify_paper_numbers.py       # 181 numeric statements of the text and every table row
python scripts/check_propositions.py         # the run's proposition checks, re-run on the CPU
python notebook/build_notebook.py --check    # the shipped notebook is rebuilt byte for byte from its cells
pytest                                       # everything above, plus notebook equivalence and file checksums
```

With the LaTeX source of the paper at hand, `python scripts/verify_paper_numbers.py --paper main.tex`
also checks that every quoted statement and every table row appears in it.

`results/v6.6/` is the paper-outputs archive of the run; `tests/test_results_integrity.py` checks every
file against the SHA-256 checksums the run recorded (`results/v6.6/manifest/file_inventory.csv`).

## 2. On a GPU: run the notebook

The notebook `notebook/convoy_ambiguity.ipynb` is self-contained: it installs what is missing, clones
the model code at pinned commits, downloads checkpoints and data, renders the synthetic testbeds,
runs every model and writes all results, tables and figures. It was developed for and run on Kaggle.

### Kaggle (as in the paper)

1. *Create → New Notebook*, then *File → Import Notebook* and choose `notebook/convoy_ambiguity.ipynb`.
2. *Settings*: accelerator **GPU T4**, **internet on**.
3. Optional: *Add-ons → Secrets* `KAGGLE_USERNAME` and `KAGGLE_KEY` (with access to the notebook) let the
   notebook copy its archives to a private Kaggle dataset. They are read at run time only.
4. *Run All*, or *Save Version → Save & Run All* for a run that must survive a closed browser.

Every expensive stage (a rendered set, a real-data preparation, optical flow, one model on one
testbed) is archived as soon as it finishes and appended to `/kaggle/working/convoy_checkpoints.zip`,
which is therefore a valid resume point at any time. The checkpointed stages alone took 10.2 h when
first computed, and downloads, installation, analyses and figures come on top, so plan two sessions:
attach the first session's `convoy_checkpoints.zip` as a dataset (it is searched under `/kaggle/input`)
and run all cells again. A
stage is restored only if its fingerprint (configuration, stage version, seed and upstream stages)
matches; otherwise it is computed again. The record of what was restored and what was computed is
`outputs/manifest/stage_log.csv`.

At the end the notebook writes `convoy_paper_outputs.zip` (figures as PDF and 300-dpi PNG with the
plotted data, tables as CSV, JSON and LaTeX, all results, manifests and logs) and keeps everything
unpacked under `/kaggle/working/outputs/`.

### Elsewhere (Colab or a local GPU)

The notebook detects Kaggle, Colab or a local machine (cell 0.3.1). Locally, install PyTorch for your
CUDA version and `pip install -r notebook/requirements-notebook.txt` (the versions of the reported
run), then start Jupyter in an empty working directory. A CUDA GPU is required (cell 0.3.3); the
largest model peaks at 5.84 GiB on the T4. Mixed precision follows the GPU: float16 below compute
capability 8.0 (the T4 of the paper), bfloat16 from 8.0 on (cell 0.3.3), so VGGT and π³ give slightly
different outputs on Ampere or newer GPUs.

| Variable | Effect |
|---|---|
| `CONVOY_ROOT` | working directory for outputs, checkpoints and scratch data (default: `/kaggle/working`, `/content`, or the current directory) |
| `CONVOY_RESUME_FROM` | a folder that holds a `convoy_checkpoints.zip` (or stage archives) to resume from |
| `CONVOY_AV2_BASE`, `CONVOY_VKITTI_BASE`, `CONVOY_BONN_BASE` | alternative download locations (mirrors) of Argoverse 2, Virtual KITTI 2 and Bonn |
| `CONVOY_KITTI_ROOT` | a local copy of KITTI tracking (optional; not part of the reported run) |

## 3. Cost of each stage

Seconds of every checkpointed stage when it was first computed on the T4 (`original_seconds` in
`results/v6.6/manifest/stage_log.csv`, each stage counted once): 90 stages, 10.2 h in total.

| Stages | Time |
|---|---:|
| Rendering CONVOY-3D: main, multi-view, flat control, E5, baseline sweep, dose-response | 56 min |
| TUM composites (CONVOY-REAL, E5-REAL) and walking sequences | 8 min |
| Optical flow (RAFT-large) on CONVOY-3D and CONVOY-REAL | 2 min |
| Virtual KITTI 2 download and ground truth | 22 min |
| Argoverse 2 ground truth, contact labels, twin heights, flow and road planes | 30 min |
| Bonn RGB-D Dynamic ground truth (not analysed in the paper) | 7 min |
| Classical baselines, oracle check | 6 min |
| Model inference, all testbeds (DUSt3R 66, MASt3R 71, MonST3R 65, VGGT 65, π³ 49, MapAnything 164 min) | 8.0 h |

Model inference without the Bonn stages, which the paper does not analyse, took 7.6 h. Downloads of
TUM and of the checkpoints, installation, the analyses and the figures come on top.

## 4. Checking a new run

Point the checks at the `outputs/` folder of a new run:

```bash
CONVOY_RESULTS=/path/to/outputs python scripts/verify_paper_numbers.py
```

The run is deterministic for a given software stack and GPU (a repeated forward pass gives
bit-identical outputs; every random stream derives from the seed 20270918). On different hardware,
drivers or library versions the model outputs, and with them the last digits of some estimates, can
differ; the verifier then lists the numbers that changed. The parts that use no network (rendering,
ground truth, classical triangulation with exact correspondences, the oracle check, the proposition
checks) do not depend on the GPU; everything that uses a model or RAFT flow does.
