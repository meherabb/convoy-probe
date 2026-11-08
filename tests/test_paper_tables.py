"""Every table row of the paper is regenerated from results/v6.6 with the paper's rounding.

The expected rows are copied verbatim from the LaTeX source of the paper (main text: Tables 1-3;
supplement: the tables of all registered contrasts and of the twin test with intervals).
"""
from convoy import paper

TABLE_1 = [
    r"P1 & $\bs$ on CONVOY-3D (predicted $0.8$--$1.2$) & 220 (159) & 0.17 & 0.19 & 0.12 & 0.09 & 0.11 & $-$0.04\\",
    r"A1 & knockout: $\bs$(full) $-$ $\bs$(mover deleted) & 89 (62) & \textbf{0.26} & \textbf{0.22} & \textbf{0.21} & \textbf{0.07} & \textbf{0.13} & 0.04\\",
    r"H5 & lifting: $\beta(\mathrm F)-\beta(\mathrm{NS})$ & 161 (71) & \textbf{0.14} & \textbf{0.31} & \textbf{0.26} & \textbf{0.11} & \textbf{0.20} & 0.08\\",
    r"H6 & hiding the contact: $\beta(\mathrm{CH})-\beta(\mathrm G)$ & 159 (71) & \textbf{0.31} & \textbf{0.39} & \textbf{0.35} & \textbf{0.26} & \textbf{0.33} & \textbf{0.14}\\",
    r"H7 & interaction: $(\beta_{\mathrm{CH}}-\beta_{\mathrm G})-(\beta_{\mathrm{FH}}-\beta_{\mathrm F})$ & 158 (71) & \textbf{0.11} & \textbf{0.32} & \textbf{0.19} & 0.06 & 0.05 & \textbf{0.12}\\",
    r"H8 & one view: $\mathrm{med}\,\rhom(\mathrm F)-\mathrm{med}\,\rhom(\mathrm{NS})$ & 194 (71) & \textbf{0.69} & \textbf{0.76} & \textbf{0.60} & \textbf{0.59} & \textbf{0.57} & \textbf{0.64}\\",
    r"H11a & real, lifting: $\beta(\mathrm F)-\beta(\mathrm G)$ & 141 (80) & \textbf{0.24} & \textbf{0.30} & \textbf{0.22} & \textbf{0.24} & $-$0.06 & \textbf{0.20}\\",
    r"H11b & real, one view: $\mathrm{med}\,\rhom(\mathrm F)-\mathrm{med}\,\rhom(\mathrm G)$ & 141 (80) & \textbf{0.21} & \textbf{0.20} & \textbf{0.21} & 0.00 & $-$0.46$^\dagger$ & \textbf{0.16}\\",
    r"H12 & baseline: $\beta(\times3)-\beta(\times0.5)$ & 148 (62) & \textbf{0.10} & \textbf{0.09} & (0.11) & (0.04) & (0.09) & (0.06)\\",
    r"H13 & 15\,cm wall: $\beta(\text{wall})-\beta(\text{no wall})$ & 145 (63) & \textbf{0.29} & \textbf{0.43} & \textbf{0.22} & \textbf{0.21} & \textbf{0.18} & \textbf{0.15}\\",
]
TABLE_1_H14 = (r"Twin minus epipolar recall of parallel vehicles with optical flow (Argoverse~2, 543 vehicles, 214 tracks): "
               r"\textbf{0.21} [0.13, 0.29]")

ALL_TESTS = [
    r"P1 & 0.17 [0.09, 0.26] & 0.19 [0.09, 0.30] & 0.12 [0.06, 0.20] & 0.09 [0.00, 0.19] & 0.11 [0.03, 0.21] & $-$0.04 [$-$0.15, 0.08]\\",
    r"A1 & 0.26 [0.11, 0.44] & 0.22 [0.07, 0.38] & 0.21 [0.07, 0.38] & 0.07 [0.04, 0.11] & 0.13 [0.02, 0.26] & 0.04 [$-$0.01, 0.10]\\",
    r"H5 & 0.14 [0.05, 0.24] & 0.31 [0.20, 0.42] & 0.26 [0.16, 0.37] & 0.11 [0.03, 0.20] & 0.20 [0.10, 0.31] & 0.08 [$-$0.04, 0.21]\\",
    r"H6 & 0.31 [0.26, 0.37] & 0.39 [0.32, 0.46] & 0.35 [0.28, 0.43] & 0.26 [0.22, 0.31] & 0.33 [0.28, 0.40] & 0.14 [0.09, 0.21]\\",
    r"H7 & 0.11 [0.05, 0.18] & 0.32 [0.24, 0.40] & 0.19 [0.12, 0.27] & 0.06 [$-$0.004, 0.13] & 0.05 [$-$0.02, 0.13] & 0.12 [0.04, 0.21]\\",
    r"H8 & 0.69 [0.60, 0.73] & 0.76 [0.72, 0.84] & 0.60 [0.55, 0.64] & 0.59 [0.56, 0.63] & 0.57 [0.53, 0.61] & 0.64 [0.59, 0.72]\\",
    r"H11a & 0.24 [0.15, 0.34] & 0.30 [0.20, 0.41] & 0.22 [0.15, 0.29] & 0.24 [0.18, 0.31] & $-$0.06 [$-$0.31, 0.16] & 0.20 [0.14, 0.27]\\",
    r"H11b & 0.21 [0.17, 0.23] & 0.20 [0.16, 0.24] & 0.21 [0.18, 0.24] & 0.00 [$-$0.05, 0.05] & $-$0.46 [$-$1.04, $-$0.13] & 0.16 [0.11, 0.21]\\",
    r"H12 & 0.10 [0.01, 0.19] & 0.09 [0.04, 0.14] & 0.11 [0.07, 0.16] & 0.04 [0.01, 0.09] & 0.09 [0.00, 0.19] & 0.06 [0.03, 0.11]\\",
    r"H13 & 0.29 [0.20, 0.38] & 0.43 [0.36, 0.51] & 0.22 [0.15, 0.30] & 0.21 [0.16, 0.26] & 0.18 [0.14, 0.24] & 0.15 [0.10, 0.19]\\",
]

TABLE_2 = [
    r"M1 epipolar residual & exact & 0.17 & 1.00\\",
    r"M1 epipolar residual & RAFT & 0.03 & 0.71\\",
    r"M2 rigid-flow residual & exact & 0.25--0.67 & 0.64--0.97\\",
    r"M2 rigid-flow residual & RAFT & 0.10--0.33 & 0.27--0.54\\",
    r"M4 model confidence & -- & 0.12--0.34 & 0.18--0.39\\",
]

# Table 3 and its supplementary version with intervals (the first column, labels/flow, is laid out with \multirow)
TABLE_3 = [
    r"parallel & 447 (195) & 0.25 & 0.91 & 0.96\\", r"intermediate & 144 (99) & 1.00 & 0.90 & 1.00\\",
    r"perpendicular & 88 (50) & 1.00 & 0.93 & 1.00\\", r"all & 679 (265) & 0.51 & 0.91 & 0.97\\",
    r"parallel & 543 (214) & 0.08 & 0.29 & 0.36\\", r"intermediate & 210 (117) & 0.61 & 0.27 & 0.77\\",
    r"perpendicular & 141 (65) & 0.93 & 0.09 & 0.95\\", r"all & 894 (288) & 0.34 & 0.25 & 0.55\\",
]
TWIN_CI = [
    r"parallel & 447 (195) & 0.25 [0.20, 0.30] & 0.91 [0.87, 0.96] & 0.96 [0.93, 0.98]\\",
    r"intermediate & 144 (99) & 1.00 [1.00, 1.00] & 0.90 [0.83, 0.96] & 1.00 [1.00, 1.00]\\",
    r"perpendicular & 88 (50) & 1.00 [1.00, 1.00] & 0.93 [0.88, 0.98] & 1.00 [1.00, 1.00]\\",
    r"all & 679 (265) & 0.51 [0.45, 0.56] & 0.91 [0.88, 0.95] & 0.97 [0.96, 0.99]\\",
    r"parallel & 543 (214) & 0.08 [0.05, 0.10] & 0.29 [0.22, 0.36] & 0.36 [0.30, 0.43]\\",
    r"intermediate & 210 (117) & 0.61 [0.53, 0.68] & 0.27 [0.19, 0.36] & 0.77 [0.70, 0.83]\\",
    r"perpendicular & 141 (65) & 0.93 [0.88, 0.97] & 0.09 [0.04, 0.14] & 0.95 [0.90, 0.99]\\",
    r"all & 894 (288) & 0.34 [0.28, 0.39] & 0.25 [0.20, 0.31] & 0.55 [0.50, 0.61]\\",
]


def test_table_1_registered_tests():
    assert paper.to_latex_rows(paper.registered_tests_table()) == TABLE_1
    h = paper.h14_row()
    assert (f"Twin minus epipolar recall of parallel vehicles with optical flow (Argoverse~2, {h['n']} vehicles, "
            f"{h['n_tracks']} tracks): {h['estimate']} {h['ci']}") == TABLE_1_H14


def test_all_registered_contrasts_with_intervals():
    assert paper.to_latex_rows(paper.all_tests_table()) == ALL_TESTS


def test_table_2_masks():
    assert paper.to_latex_rows(paper.masks_table()) == TABLE_2


def test_table_3_twin_test():
    rows = paper.twin_table()
    assert [r[0] for r in rows] == ["labels"] * 4 + ["flow"] * 4
    assert paper.to_latex_rows([r[1:] for r in rows]) == TABLE_3
    assert paper.to_latex_rows([r[1:] for r in paper.twin_table(intervals=True)]) == TWIN_CI


def test_markdown_tables_render():
    md = paper.to_markdown(["Test", "Quantity", "n"] + [paper.MODEL_NAMES[m] for m in paper.MODELS],
                           paper.registered_tests_table(latex=False))
    assert md.count("\n") == 11 and "**0.39**" in md and "(0.11)" in md and "−0.46†" in md
