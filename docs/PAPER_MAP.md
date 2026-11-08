# From the paper to the code and the outputs

Where each part of the paper is computed (notebook section and cell source in
`notebook/src/cells/`), where its numbers are stored (paths relative to `results/v6.6/`), and where
the repository checks it. Every number of the text is listed with its source in
`src/convoy/claims.py`, every table row in `src/convoy/paper.py`.

**Numbering.** Figure and table numbers below are those of the paper. The figure *files* keep the
names the notebook gave them (`fig3_geometry_share.png` is Figure 2 of the paper; see the table at
the end). The notebook numbers four propositions, the paper three: notebook Propositions 1 and 2
are the paper's Propositions 1 and 2, notebook Proposition 3 (only motion normal to the epipolar
plane is visible) is a statement in Sec. 3, and notebook Proposition 4 (the static twin) is the
paper's Proposition 3. The keys `P1_…`–`P4_…` of `logs/proposition_checks.json` follow the
notebook; "P1"–"P4" in the tables of the paper are the locked *predictions*.

## Theory (Sec. 3, Supp. A)

| Paper | Notebook | Outputs | Repository |
|---|---|---|---|
| Proposition 1, parallel motion | 2.1 `0310_geometry_core.py`, 2.2 `0320_prop_checks.py` | `logs/proposition_checks.json` (`P1_*`), `logs/testbed_checks.json` | `convoy.geometry.static_twin`, `tests/test_geometry.py`, `scripts/check_propositions.py` |
| Proposition 2, several views | 2.2 | `P2_*` | `tests/test_geometry.py` |
| Motion inside the epipolar plane | 2.2 | `P3_*` | `tests/test_geometry.py` |
| Proposition 3, static twin, Eq. (1) | 2.2 | `P4_*` | `convoy.geometry.twin_contact_height` |
| Lemma 1, slope and weight | 2.4 in `0330_a2_identity.py` | `logs/a2_and_identity.json` | `convoy.probe.mixture_rho`, `slope_weight_table`, `tests/test_probe.py` |
| The probe: scale alignment, ρ, regimes, sample rules | 4.6 `0680_measurement.py`, 3.2 `0470_render_all.py` | `manifest/convoy3d_*_movers.csv` | `convoy.probe`, `tests/test_probe.py` |
| Per-object share g | 6.8b `0887b_x6.py` | – | `convoy.probe.per_object_share` |

## Experimental design (Sec. 4, Supp. B)

| Paper | Notebook | Outputs |
|---|---|---|
| Models, Table 4 | 4.1–4.5 `0610_repos.py` … `0670_registry.py` | `tables/T1a_models.csv`, `logs/model_status.csv`, `manifest/repos.json` |
| Data, Table 5 | 3.1–3.12 | `tables/T1b_data.csv`, `manifest/datasets.json`, `manifest/{vkitti,argoverse2,bonn}.json` |
| Validity gate (Fig. 5d) | 6.0 `0820_gate.py` | `results/gate.csv`, `tables/T2_validity_gate.csv` |
| Estimator check (Fig. 5e, Supp. A) | 6.0b `0825_oracle.py` | `results/oracle_validation.csv` |
| Statistics | 6.A `0810_stats.py` | `convoy.stats` |
| Registration | 0.1 `0010_idea_lock.py`, 0.2 `0030_decisions.py` | [`registration/`](../registration/README.md) |

## Results (Sec. 5)

| Paper | Notebook | Outputs |
|---|---|---|
| Table 1 P1, Fig. 2a | 6.1 `0830_e1.py` | `results/E1_beta_s.csv`, `tables/T3_E1_geometry_share.csv` |
| Table 1 A1, Fig. 2b | 6.5 `0870_a1.py` | `results/A1_knockout.csv` |
| Single-image-corrected slopes (Sec. 5.1, Supp. D) | 6.1, 6.5 | `results/E1_beta_s.csv` (`beta_s`, `beta_mono`), `results/A1_knockout.csv` (`beta_full`, `beta_knockout`, `beta_mono`); the differences are computed in `convoy.claims` |
| Fig. 2c, share by parallax | 6.1 | `results/E1_beta_by_parallax.csv` |
| Grounded and floating movers (Sec. 5.1) | 6.9 `0889_synth.py` | `results/support_levels.csv` |
| Table 1 H5–H8, Fig. 3 | 3.1.7 `0445_ablation.py`, 6.7 `0884_e5.py` | `results/E5_contrasts.csv`, `results/E5_beta_by_condition.csv` |
| Table 1 H11a–b, Fig. 7 | 3.5.1 `0535_real_contact.py`, 6.7b `08852_e5r.py` | `results/E5real_contrasts.csv`, `results/E5real_beta_by_condition.csv` |
| Table 1 H12, Fig. 6b–c | 3.2b `0472_designed.py`, 6.10 `0889b_designed.py` | `results/BSW_tests.csv`, `results/BSW_beta.csv` |
| Table 1 H13, Fig. 6d | 3.2b, 6.10 | `results/WDR_tests.csv`, `results/WDR_beta.csv`, `tables/T20a_dose_response.csv` |
| CONVOY-REAL (Sec. 5.4) | 3.5 `0530_convoy_real.py`, 6.6 `0880_real.py` | `results/real_E1.csv`, `tables/T21e_convoy_real_by_parallax.csv` |
| Argoverse 2 slopes (Sec. 5.4) | 3.11 `0578_av2.py`, 3.11b `0579_av2x.py`, 6.8b `0887b_x6.py` | `results/real_driving_beta.csv`, `results/X6_contact*.csv` |
| All registered tests, Table 6 | 7 `0900_summary.py` | `results/prediction_summary.csv` |

## Consequences (Sec. 6)

| Paper | Notebook | Outputs | Repository |
|---|---|---|---|
| Table 2, Fig. 9a | 6.2 `0840_e2.py` | `results/E2_masks.csv`, `results/E2_recall_by_eps.csv` | `convoy.masks` |
| Fig. 9b, masks by support | 6.7c `0884b_e5masks.py`, 6.11 `0889d_cue.py` | `results/E5_masks_by_condition.csv`, `results/mask_tradeoff.csv` | – |
| Prevalence (Sec. 6.2, Fig. 10b) | 3.7 `0550_real_dynamic.py`, 6.8 `0887_e6.py`, 6.8b | `results/prevalence_*.csv`, `results/real_driving_prevalence.csv` | – |
| Convoy case (Sec. 6.2, Fig. 10c) | 6.8b | `results/X6_pace.csv` | – |
| Twin test: Table 3, Table 7, Fig. 4 | 3.11c `0579c_twin.py`, 3.11d `0579d_flowtwin.py`, 6.13 `0889h_twin.py` | `results/twin_*.csv`, `results/twin_*thresholds.json` | `convoy.twin`, `tests/test_notebook_equivalence.py` |
| Vehicles without a static twin (Sec. 6.3, Supp. D) | 6.13 | `results/twin_thresholds.json` and `results/twin_flow_thresholds.json` (`n_moving`, `n_no_static_solution`) | `convoy.twin.run_label_test`, `flow_static_solution` |
| H14 | 6.13 | `results/twin_flow_thresholds.json` (`H14`) | `convoy.twin.h14_contrast` |

## Discussion and limitations (Sec. 7)

| Paper | Notebook | Outputs |
|---|---|---|
| MonST3R against DUSt3R on the same movers | 6.11 `0889d_cue.py`, 11 `1100_tables.py` | `tables/T21d_monst3r_vs_dust3r.csv` |
| H9 on Virtual KITTI 2 (exploratory) | 6.8 `0887_e6.py` | `results/E6_vkitti_h9.csv`, `tables/T10d_vkitti_sample.csv` |
| H10, withdrawn | 0.1 `0010_idea_lock.py` | `manifest/amendment_v6_1.json` (`withdrawn`) |
| Bonn RGB-D Dynamic, objects per model | 6.12 `0889f_bonn.py` | `results/bonn_*.csv` |

## Supplement (Supp. A–F)

| Paper | Notebook | Outputs |
|---|---|---|
| Numerical checks, renders (Supp. A) | 2.2, 3.3 | `logs/proposition_checks.json`, `logs/testbed_checks.json` |
| Inputs and run (Supp. B) | 0.3.4 `0080_seeds.py`, 13 `1300_repro.py` | `logs/env_log.json`, `manifest/stage_log.csv` |
| Baseline sweep, single-image check (Supp. B) | 6.10 | `results/BSW_beta.csv` (`median_rho_mono`) |
| Locked predictions P2–P4 | 6.2, 6.3 `0850_e3.py`, 6.4 `0860_e4.py` | `results/E2_masks.csv`, `results/E3_prior_weight.csv`, `results/E4_summary.csv` |
| H9 and H10 (Supp. C) | 6.8, 0.1 | `results/E6_vkitti_h9.csv`, `tables/T10d_vkitti_sample.csv`, `manifest/amendment_v6.json`, `manifest/amendment_v6_1.json` |
| Dose-response, single-image error (Supp. D) | 6.10 | `results/WDR_beta.csv` (`median_rho_mono`) |
| Fig. 11, prior weight and views; 2 to 8 frames (Supp. D) | 6.3, 6.4 | `results/E3_w_by_eps.csv`, `results/E4_multiview.csv`, `results/E4_summary.csv` |
| Fig. 8, support across testbeds | 6.9 | `results/support_levels.csv`, `results/support_contrasts.csv` |
| Contact in traffic | 6.8b | `results/X6_contact_test.csv`, `results/X6_contact_matched.csv` |
| Robustness | 8 `0910_robust.py` | `results/robustness.csv` |
| Broader impact (Supp. E), galleries (Supp. F) | 10 | `figures/figS2_gallery.png`, `figures/figS4_av2_contact_twin.png`, `figures/figS5_designed_gallery.png` |

## Figures

| Paper | File in `results/v6.6/figures/` | Notebook cell |
|---|---|---|
| Figure 1 | `fig1_teaser.png` | `1010_fig1.py` |
| Figure 2 | `fig3_geometry_share.png` | `1030_fig3.py` |
| Figure 3 | `fig4_ablation.png` | `1040_fig4.py` |
| Figure 4 | `fig11_static_twin.png` | `1086_fig11.py` |
| Figure 5 | `fig2_testbeds_checks.png` | `1020_fig2.py` |
| Figure 6 | `fig13_designed.png` | `1088_fig13.py` |
| Figure 7 | `fig5_real_contact.png` | `1045_fig5.py` |
| Figure 8 | `fig10_support.png` | `1085_fig10.py` |
| Figure 9 | `fig6_masks.png` | `1050_fig6.py` |
| Figure 10 | `fig7_driving.png` | `1060_fig7.py` |
| Figure 11 | `fig8_prior_multiview.png` | `1070_fig8.py` |
| Figure 12 | `figS2_gallery.png` | `1090_figS.py` |
| Figure 13 | `figS5_designed_gallery.png` | `1091_figS45.py` |
| Figure 14 | `figS4_av2_contact_twin.png` | `1091_figS45.py` |
| not in the paper | `fig9_qualitative.png`, `fig12_two_channels.png`, `fig14_bonn.png`, `figS1_a2_first_order.png`, `figS3_m5_remedy.png` | `1080_fig9.py`, `1087_fig12.py`, `1089_fig14.py`, `1090_figS.py` |

The captions the notebook wrote are in `manifest/figures.json`; the paper's captions were written
for the paper and differ in wording (see [results/v6.6/README.md](../results/v6.6/README.md) for one
correction to a notebook caption).

## In the notebook but not interpreted in the paper

Bonn RGB-D Dynamic beyond its object counts (3.12, 6.12; too few objects met the sample rules), the
two-channel cue model (6.11; only its MonST3R-against-DUSt3R table enters the paper), the M5 mask,
assumption A2 of the idea lock (2.3), the failure analysis (9) and KITTI tracking (needs a private
copy; not part of the run). The claim audit of the notebook (12 `1200_claims.py`,
`results/claim_audit.csv`) predates the paper; the paper's own numbers are those checked by
`scripts/verify_paper_numbers.py`.
