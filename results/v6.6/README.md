# Outputs of the reported run (notebook v6.6)

This folder is the paper-outputs archive (`convoy_paper_outputs.zip`) of the notebook run reported in
the paper: Kaggle, one Tesla T4, Python 3.13, PyTorch 2.11 with CUDA 12.8, seed 20270918
(`logs/env_log.json`). The files are unchanged: `manifest/file_inventory.csv` lists the size and
SHA-256 checksum the run recorded for every file, and `tests/test_results_integrity.py` checks all of
them. Not included are the vector PDFs of the figures with their plotted data, the JSON and LaTeX
renderings of the tables, and the per-model measurement cache; they are rebuilt by every run of the
notebook.

| Folder | Content |
|---|---|
| `results/` | unformatted results of every analysis: estimates, intervals, sample sizes, verdicts, bins (64 files) |
| `tables/` | the formatted tables of the notebook as CSV (47 tables; captions in `manifest/tables.json`) |
| `figures/` | the 19 figures of the notebook as 300-dpi PNG (captions in `manifest/figures.json`) |
| `manifest/` | registration record, run configuration, model, data and stage status, per-mover ground truth of the synthetic and composite testbeds, stage registry (`stages/`), RAM log, file checksums |
| `logs/` | environment, installation, proposition checks, testbed self-checks, ground-truth validation (A3), model status |

## The files most readers need

| Question | File |
|---|---|
| Every registered and locked test, with estimate, interval, Holm p-value and verdict | `results/prediction_summary.csv` |
| Geometry share of each model on CONVOY-3D (Table 1, P1) | `results/E1_beta_s.csv` |
| Contact ablation E5 (H5–H8) and its real-image version (H11) | `results/E5_contrasts.csv`, `results/E5real_contrasts.csv` |
| Baseline sweep (H12) and dose-response (H13) | `results/BSW_tests.csv`, `results/WDR_tests.csv` |
| Dynamic masks (Table 2) | `results/E2_masks.csv` |
| Static-twin test and H14 (Table 3) | `results/twin_recall_by_regime.csv`, `results/twin_flow_recall_by_regime.csv`, `results/twin_flow_thresholds.json` |
| Validity gate and oracle check | `results/gate.csv`, `results/oracle_validation.csv` |
| Single-image slopes of the E1 and A1 movers (the corrected slopes of Sec. 5.1) | `results/E1_beta_s.csv`, `results/A1_knockout.csv` |
| MonST3R against DUSt3R on the same movers (Sec. 7) | `tables/T21d_monst3r_vs_dust3r.csv` |
| H9 on Virtual KITTI 2 and its attrition (Supp. C) | `results/E6_vkitti_h9.csv`, `tables/T10d_vkitti_sample.csv` |
| Run configuration, every registered constant | `manifest/run_config.json` |
| What each stage cost and whether it was computed or restored | `manifest/stage_log.csv` |

The paper's figures use the PNG files of this folder; the paper numbers its figures differently from
the file names ([docs/PAPER_MAP.md](../../docs/PAPER_MAP.md) has the correspondence).

## How the run was made

The v6.6 session restored from the checkpoint archive of an earlier session every stage whose
fingerprint was unchanged and computed the others: the designed sets (baseline sweep and
dose-response) with their model evaluations, the Bonn ground truth with the six models on it, and the
two static-twin stages. `manifest/stage_log.csv` records for every stage whether it was restored or
computed and how long it took when it was computed.

## Notes

* **Caption of `figures/fig9_qualitative.png`** (not used in the paper). The notebook's caption in
  `manifest/figures.json` says that the parallel mover in panel (a) "leaves no geometric residual, so
  M2 cannot flag it". That holds for the epipolar residual M1, not for M2: M2 compares the observed
  correspondence with the flow implied by the model's depth, and DUSt3R places this mover near its
  true depth rather than at the depth of its static twin, so M2 does flag it, as the figure shows
  (Sec. 6.1 of the paper).
* The notebook also contains analyses that the paper does not interpret: Bonn RGB-D Dynamic
  (`results/bonn_*.csv`, `figures/fig14_bonn.png`), the cue model (`results/cue_*.csv`,
  `figures/fig12_two_channels.png`), the M5 mask (`results/E2_M5_by_x.csv`, `figures/figS3_m5_remedy.png`),
  assumption A2 (`logs/a2_diagnostic.npz`, `figures/figS1_a2_first_order.png`) and the failure analysis
  (`results/failure_covariates.csv`, `results/largest_deviations.csv`).
* `results/claim_audit.csv` is the notebook's own audit of the claims it anticipated; the numbers of
  the paper are those checked by `scripts/verify_paper_numbers.py`.
* Absolute paths inside the manifests (`/kaggle/working/...`) are those of the Kaggle session.
* `logs/a2_diagnostic.npz` stores the camera kind of each configuration as an object array (strings),
  which NumPy loads only with `allow_pickle=True`; its numeric arrays load without it.
