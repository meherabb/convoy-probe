# The notebook

`convoy_ambiguity.ipynb` (version 6.6) is the complete experiment: it produced every number, table
and figure of the paper. It is shipped without outputs; the outputs of the reported run are in
[`results/v6.6/`](../results/v6.6/README.md). How to run it: [docs/REPRODUCE.md](../docs/REPRODUCE.md).

## Structure

| Section | Cells | Content |
|---|---|---|
| 0 | `0000`–`0105` | idea lock and amendments, decisions, environment and paths, installation, GPU and precision policy, seeds and determinism check, run configuration with every registered constant, stage checkpoints and resume |
| 2 | `0310`–`0330` | geometry core, numerical checks of the propositions, assumption A2 and the slope-weight identity |
| 3 | `0410`–`0581` | testbeds: textures, scenes and ray casting of CONVOY-3D, the clip sampler, the E5 ablation, the designed sets, testbed self-checks; TUM, CONVOY-REAL, E5-REAL, optical flow, TUM walking, Virtual KITTI 2, LiDAR ground truth, KITTI tracking (optional), Argoverse 2, static twin and flow twin, Bonn |
| 4 | `0610`–`0690` | model code at pinned commits, adapters for the six models, registry, the measurement protocol, classical baselines |
| 5 | `0710`–`0720` | master evaluation loop (one checkpointed stage per model and testbed) and collection |
| 6 | `0810`–`0889h` | statistics, validity gate, oracle check, E1–E5, E5-REAL, protocol check, driving, synthesis, designed experiments, cue model, Bonn, twin test |
| 7–9 | `0900`–`0920` | summary of every registered test, robustness, failure analysis |
| 10 | `1000`–`1091` | figures |
| 11–13 | `1100`–`1300` | tables, claim audit, reproducibility record and archives |

Each cell starts with its number (`# Cell No: 001`) and a header naming its section.

## Building from the cell sources

The notebook is generated from `src/cells/` (one file per cell, sorted by name; `.py` code cells,
`.md` markdown cells):

```bash
python notebook/build_notebook.py            # writes notebook/build/convoy_ambiguity.ipynb
python notebook/build_notebook.py --check    # and compares it byte for byte with the shipped notebook
```

The builder numbers the cells, checks that every code cell compiles and that the notebook is valid
nbformat 4.5, that no figure text is set in bold, that it stays below Kaggle's 1 MB upload limit, and
that no cell contains an e-mail address or any term listed in the environment variable
`NOTEBOOK_DENYLIST` (comma-separated), which keeps identifying strings out of an anonymous release.

To change the experiment, edit the cell sources, rebuild, and run the new notebook. A stage whose
configuration or version (`STAGE_VERSIONS` in cell 0.4) changes is recomputed; all others are restored
from the checkpoint archive.

The model-independent functions of the notebook (geometry, measurement, statistics, the twin test)
also exist as the tested package `src/convoy`; `tests/test_notebook_equivalence.py` runs both on the
same inputs and requires identical results.

`requirements-notebook.txt` lists the Python packages of the reported run for a local GPU run; on
Kaggle the notebook installs what is missing by itself.
