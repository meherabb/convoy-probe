# PLACEHOLDER
# =============================================================================
# 0.1  IDEA LOCK AND V6 AMENDMENT.
#      IDEA_LOCK: the scientific content locked before the first experiment
#      (question, hypothesis, falsification conditions, propositions,
#      assumptions, predictions P1-P4, metrics, claim scope), unchanged. Only the
#      two "actually_*" inventory fields record what THIS notebook version runs.
#      AMENDMENT_V6: the reframing and the hypotheses H5-H10, registered after
#      the v5 results were read and before any v6 experiment was generated.
#      AMENDMENT_V6_1: written after the first complete v6 run and before any
#      v6.1 data: H11a-b (contact ablation on real images) registered, H10
#      withdrawn, H9 moved to exploratory. H5-H8 and P1-P4 are unchanged.
#      AMENDMENT_V6_2 to AMENDMENT_V6_4: exploratory analyses only, each written
#      after reading the previous complete run and before its own data existed.
#      AMENDMENT_V6_5: two designed experiments with registered predictions
#      (H12 baseline sweep, H13 contact dose-response) and exploratory analyses
#      (static twin, cue model, Bonn RGB-D Dynamic), written after the complete
#      v6.4 run and before any of their data existed.
#      AMENDMENT_V6_6: H14 (the twin test with optical flow) registered, the Bonn
#      ground truth corrected, written after the complete v6.5 run.
#      All of them are written to outputs/manifest/ in Section 13.
# =============================================================================
IDEA_LOCK = {
    "title": "The Convoy Ambiguity: Feed-Forward 3D and 4D Reconstruction Models "
             "Inherit the Epipolar Degeneracy for Motion Parallel to Camera Translation",

    "contribution_type": "Scientific discovery (primary); Evaluation (motion-decomposed "
        "protocol for dynamic-scene reconstruction); an elementary, exact geometric law "
        "fixes the falsifiable prediction (not a general theory paper).",

    "research_question": (
        "Do frozen feed-forward 3D/4D reconstruction models (DUSt3R, VGGT, MonST3R, ...) "
        "reconstruct an object moving parallel to the camera's translation at the exact "
        "depth a static object at that location would have (Z*(s)=Z/(1-s)), and do "
        "training-free dynamic-mask methods built on epipolar-inconsistency fail to flag "
        "exactly this motion?"
    ),

    "central_hypothesis": (
        "For static-trained feed-forward models, the predicted depth of an object moving "
        "with parallel displacement fraction s and perpendicular pixel offset eps~=0 "
        "satisfies Zhat/Z = 1/(1-s) to within the model's static-scene error, with no "
        "increase in any dynamic-mask signal; as eps grows, the prediction departs from "
        "the epipolar solution toward the model's monocular estimate; dynamic-trained "
        "models differ from static-trained ones only in the eps>0 regime and, at eps~=0, "
        "deviate from 1/(1-s) only for object classes with strong size priors."
    ),

    "falsification_conditions": (
        "The hypothesis is FALSIFIED if: (a) Zhat/Z is insensitive to s (priors dominate "
        "everywhere -> beta_s CI excludes [0.5,1.5] or R^2<0.5 at eps~=0); "
        "(b) dynamic-mask methods flag parallel motion (|eps|<1px) as reliably as "
        "perpendicular motion (eps>=3px) -> recall-difference CI includes 0; or "
        "(c) dynamic-trained models beat Z*(s) at eps~=0 for ALL object classes, i.e. no "
        "class-dependent prior takeover is observed."
    ),

    "propositions_exact_and_verified": {
        "Proposition 1 (parallel-motion degeneracy)": (
            "delta = s*b (pure parallel motion) makes the moving point's camera-2 "
            "coordinates identical, up to a positive scalar, to the static point "
            "X* = X/(1-s)'s camera-2 coordinates -> both pixels are indistinguishable "
            "from a static point at depth Z*(s) = Z/(1-s). Verified numerically in "
            "Section 2 to floating-point precision, for arbitrary camera rotation R."
        ),
        "Proposition 2 (multi-view convoy persistence)": (
            "If the camera's TRANSLATION per frame is constant (b each frame -> a "
            "straight path) and the object's own world-frame velocity is s*b (constant, "
            "parallel to that translation), the degeneracy persists for any number of "
            "frames and for ANY per-frame camera ROTATION R_k. It requires a CHANGE in "
            "the camera's translation direction (a curved path / true turning) to break -- "
            "rotation of the camera's heading alone, with a straight translation path, "
            "does NOT break it. This distinction is verified numerically in Section 2."
        ),
        "Proposition 3 (perpendicular motion is detectable)": (
            "If the perpendicular component delta_perp != 0, no static point reproduces "
            "both observations; the second observation lies eps pixels off the true "
            "epipolar line -- the only first-order geometric evidence of motion available "
            "to a two-view method."
        ),
    },

    "assumptions": {
        "A1 (correspondence-based depth)": (
            "The model's two-view depth for textured object pixels is driven by cross-view "
            "correspondence, not monocular cues alone. DIRECTLY TESTABLE per model via the "
            "static-scene sensitivity test (mask the corresponding patch in view 2, observe "
            "the depth change). If it fails for a model, w~=1 is EXPECTED for that model and "
            "is reported as such, not treated as a hypothesis failure."
        ),
        "A2 (small-displacement first-order geometry)": (
            "eps computed to first order (eps ~= f*||delta_perp||/Z) is adequate. "
            "DIRECTLY TESTABLE by comparing the first-order formula against the exact "
            "epipolar-line perpendicular distance on synthetic data with known geometry. "
            "Where it is violated, the EXACT offset is used instead for regime "
            "classification -- see Section 2.3 for the measured divergence."
        ),
        "A3 (ground-truth adequacy)": (
            "Synthetic data supplies exact depth/velocity by construction; any real data "
            "(Sintel, TUM) supplies GT depth/flow/poses with bounded, documented sensor "
            "noise. STRUCTURAL for the synthetic testbed (true by construction); "
            "EMPIRICALLY PROXY-TESTABLE for any real dataset that loads, via GT "
            "self-consistency checks."
        ),
    },

    "falsifiable_numerical_predictions": {
        "Prediction 1 (degeneracy)": "beta_s in [0.8,1.2] (95% cluster-bootstrap CI) for "
            "static-trained models in the parallel regime (|eps|<1px), R^2>=0.5.",
        "Prediction 2 (masks)": "Dynamic-mask recall for |eps|<1px is lower than for "
            "eps>=3px by >=0.3 absolute, at matched |s| and object size.",
        "Prediction 3 (prior takeover)": "Prior weight w increases with eps (Spearman "
            "rho>0.5) and, at eps~=0, is larger for size-prior-bearing classes than for "
            "texture-only synthetic objects.",
        "Prediction 4 (convoy persistence)": "beta_s stays in [0.8,1.2] over 2-8 frames on "
            "straight (constant-translation) segments and drops toward 0 on segments with "
            "a genuinely curved translation path.",
    },

    "method_under_test": "Feed-forward 3D/4D reconstruction models, probed by the "
        "measurement protocol in Section 4 (not a newly proposed architecture).",

    "models_in_scope_per_idea": [
        "DUSt3R", "MASt3R", "VGGT-1B", "pi3", "MapAnything",
        "Depth Anything 3", "MonST3R", "CUT3R",
    ],
    "models_actually_registered_this_notebook": [
        "DUSt3R", "MASt3R", "MonST3R", "VGGT-1B", "pi3", "MapAnything",
    ],

    "datasets_in_scope_per_idea": [
        "Kubric MOVi-F", "MPI-Sintel", "KITTI tracking",
        "Bonn RGB-D Dynamic", "TUM RGB-D dynamic",
    ],
    "datasets_actually_used_this_notebook": [
        "CONVOY-3D (ray-cast synthetic street scenes; plays the role of Kubric MOVi-F, Section 3.1), "
        "including the v6 ground-contact ablation (E5)",
        "LEGACY-FLAT (negative control for the validity gate, Section 3.4)",
        "TUM RGB-D fr3 static sequences (real backgrounds for the CONVOY-REAL composites, Section 3.5, and "
        "for the v6.1 contact ablation on real images, E5-REAL, Section 3.5.1)",
        "TUM RGB-D fr3 walking_xyz / walking_halfsphere (prevalence of motion regimes only, Section 3.7)",
        "Virtual KITTI 2, clone variation, Camera_0 (photo-realistic driving with exact depth and poses, Section 3.8)",
        "Argoverse 2 sensor dataset, validation logs, front-centre camera (real driving with LiDAR; exploratory, Sections 3.11-3.11c)",
        "KITTI tracking (real driving with LiDAR; only when attached; exploratory, Section 3.10)",
        "CONVOY-3D designed experiments of v6.5: baseline sweep (BSW) and contact dose-response (WDR), Section 3.2b",
        "Bonn RGB-D Dynamic, four sequences with balloons and boxes moved by people (real objects with measured depth; "
        "exploratory, Section 3.12)",
    ],

    "baselines_per_idea": [
        "Classical two-view triangulation with GT poses (positive control -- MUST follow "
        "1/(1-s) exactly)",
        "Monocular model / oracle (negative control -- MUST be s-invariant)",
        "Flow-based dynamic mask (MonST3R-style)",
        "Attention-based dynamic mask (Easi3R-style / VGGT4D-style)",
    ],

    "main_metrics": ["beta_s (regression slope)", "w(s,eps,class,texture) (prior weight)",
                     "dynamic-mask recall by regime", "depth AbsRel by motion angle"],

    "sota_position": "Scientific discovery + evaluation advance. The paper predicts a "
        "FAILURE of existing dynamic-mask methods rather than beating a leaderboard. "
        "No SOTA is claimed anywhere in this notebook.",

    "claim_scope": (
        "Findings are scoped to: (i) the specific models whose checkpoints actually "
        "loaded in THIS execution (see MODEL_STATUS in Section 4); (ii) the datasets that "
        "actually became available (see DATASET_STATUS in Section 3); (iii) the parameter "
        "ranges swept (depth 5-60 m, |s| up to 1.4, frame gaps and baselines documented in "
        "Section 0.2). No claim in this notebook generalizes beyond what is listed here."
    ),
}

# =============================================================================
# V6 AMENDMENT (registered before any v6 data were generated; the v5 run had shown
# beta_s between 0 and 0.4 for every model, i.e. P1 falsified as locked)
# =============================================================================
AMENDMENT_V6 = {
    "status": "registered after reading the v5 results and before generating any v6 data; "
              "IDEA_LOCK and the verdict rules of P1-P4 are not changed",

    "reframing": (
        "Parallel motion is used as an exact counterfactual. For a mover with delta = s*b the two "
        "views are pixel-identical to a static object at Z/(1-s) (Proposition 1), so the geometric "
        "answer and the true depth differ by the known amount x = -log(1-s) while nothing else in the "
        "images changes. The slope beta_s of the log depth error on x therefore measures how much of "
        "the cross-view geometry a model uses for the object: 1 = pure triangulation, 0 = pure single-"
        "image prior. Through the slope-weight identity (Section 2.4) beta_s maps to a prior weight, "
        "and Section 6.0b checks that the estimator recovers planted weights. The question becomes: "
        "how much geometry do feed-forward models use for moving objects, what controls it, and what "
        "does that imply for dynamic masks built on geometric residuals?"
    ),

    "registered_hypotheses": {
        "H5 (support cue)": "Ground-contact ablation E5 (same object, same s, eps = 0): beta(floating, no "
            "shadow) - beta(grounded, no shadow) > 0, paired cluster-bootstrap 95% CI > 0, per model, car and "
            "crate pooled. Rationale: without a ground contact the single-image depth cue of the object is "
            "missing, so a model that combines cues should lean more on geometry.",
        "H6 (contact visibility)": "beta(grounded, contact hidden by a wall) - beta(grounded) > 0, CI > 0.",
        "H7 (the wall acts through the contact)": "(beta_CH - beta_G) - (beta_FH - beta_F) > 0, CI > 0: hiding "
            "the contact matters more for an object that has one.",
        "H8 (single-image signature)": "median rho_mono(floating) - median rho_mono(grounded, no shadow) > 0, "
            "CI > 0: a lifted object is placed farther away by the single-image prediction (its lowest "
            "pixels sit higher in the image, as for a farther grounded object).",
        "H9 (driving)": "Virtual KITTI 2 moving vehicles in the parallel regime (|eps| < 1 px), measurable and "
            "identifiable, x_geo = log(Z_geo/Z) inside the CONVOY-3D design range [-0.47, 1.90]: the slope "
            "beta_geo of rho on x_geo is below 0.3. SUPPORTED if the estimate is < 0.3 and the CI upper end "
            "< 0.5; FALSIFIED if the CI lower end > 0.3; PARTIAL otherwise. Tested only with >= 20 movers "
            "from >= 5 pairs. Models trained on Virtual KITTI (MASt3R, VGGT) are flagged, not removed.",
        "H10 (attention masks on long clips)": "Easi3R on 10-frame CONVOY-3D clips (window-5 scene graph, "
            "the dynamic map of Eq. 9 refined by k-means clustering of encoder features as in the Easi3R "
            "paper), evaluated on the middle frame: recall(perpendicular) - recall(parallel) >= 0.3 with "
            "CI > 0, using the P2 rule unchanged (recall at 5% static FPR, stratified by |s| and size). "
            "Backbones DUSt3R and MonST3R. Replicated on Virtual KITTI 2 clips when each regime has >= 20 movers.",
    },

    "exploratory_without_prediction": [
        "E5 shadow effect: beta(grounded, no shadow) - beta(grounded).",
        "E5 class effect: floating car versus floating crate.",
        "E6 prevalence: share of moving-vehicle pixels in Virtual KITTI 2 within 1 px of the epipolar line, and "
        "share of moving vehicles whose track over K = 2, 4, 8 frames a static point fits within 1 px.",
        "E7 secondary outputs: the raw Eq. 9 map and the Otsu-thresholded binary Easi3R mask.",
    ],

    "removed_from_v5": [
        "Real-people slope analysis (v5 Section 6.7): the regressor log(Z_tri/Z) is built from noisy flow and "
        "sensor depth, which attenuates the slope by an unknown factor. Only the prevalence of motion "
        "regimes in TUM walking is kept.",
        "M3 Easi3R on a 3-frame complete graph (v5 E2): far shorter than the windows the method was designed "
        "for. Replaced by H10 on 10-frame clips.",
    ],
}

# =============================================================================
# V6.1 AMENDMENT (written after the first complete v6 run, before any v6.1 data)
# =============================================================================
AMENDMENT_V6_1 = {
    "status": "written after reading the first complete v6 run and before generating any v6.1 data "
              "(the real-image contact ablation did not exist in v6). H5-H8 and P1-P4 keep their rules.",

    "registered_hypotheses": {
        "H11a (support cue, real images)": "E5-REAL: the same textured crate inserted into real TUM RGB-D frame "
            "pairs, once resting on the real support plane with a contact shadow (G) and once lifted 0.25-0.50 m "
            "without a shadow (F); purely parallel motion delta = s*b. beta(F) - beta(G) > 0, paired cluster-"
            "bootstrap 95% CI > 0 (cluster = image pair), Holm across models. Same estimator as H5.",
        "H11b (single-image signature, real images)": "median rho_mono(F) - median rho_mono(G) > 0, CI > 0, on the "
            "same crates. Same estimator as H8.",
        "sample": "crates measurable and identifiable in both conditions, with >= 5 contact pixels (within "
            "3 cm of the plane) in both views when grounded and none when lifted; at least 8 crates from 5 pairs, "
            "otherwise NOT TESTABLE.",
    },

    "withdrawn": {
        "H10 (Easi3R on 10-frame clips)": "Outcome of the first complete v6 run, kept on record: FALSIFIED by its "
            "rule for both backbones (recall(perp) - recall(par) = -0.02 [-0.15, 0.10] for DUSt3R and -0.08 "
            "[-0.18, 0.02] for MonST3R, 52 parallel and 43 perpendicular movers). The rule could not have come out "
            "otherwise: the cluster-refined map found perpendicular movers with recall 0.17 and 0.07 at a 5% static "
            "false-positive rate, and the binary mask flagged 40-45% of static pixels, so the map detects movers in "
            "neither regime and cannot show a regime difference. The test is withdrawn from v6.1 and the attention-"
            "based mask of the locked baseline list is not evaluated (Section 0.2, D33); the paper reports this in "
            "its limitations, not as evidence for or against the probe.",
    },

    "moved_to_exploratory": {
        "H9 (driving)": "The v6.0 measurability rule for Virtual KITTI 2 (whole box inside the crop, truncation "
            "<= 5%, >= 85% visible in view 2) left 2 vehicles, so H9 could not be tested. v6.1 applies the "
            "CONVOY-3D pixel rule (>= 40 eroded visible pixels, >= 85% of the silhouette unoccluded). Because "
            "the rule was changed after the v6.0 sample size was seen, H9 is reported as exploratory, with "
            "its registered thresholds and the full attrition table (Section 3.8), not as a confirmatory test.",
    },
}

# =============================================================================
# V6.2 AMENDMENT (written after the complete v6.1 run, before any v6.2 analysis existed)
# =============================================================================
AMENDMENT_V6_2 = {
    "status": "written after reading the complete v6.1 run and before any of the analyses below existed. Nothing in it "
              "is a registered test: every item is exploratory and none changes a registered verdict.",
    "exploratory_analyses": {
        "masks by support (6.7c)": "M2 recall (exact correspondences) on the same parallel E5 movers in each condition; "
            "expected direction: lower when the contact is hidden (G -> CH) or the mover lifted (NS -> F).",
        "ground contact in traffic (6.8b)": "Virtual KITTI 2 parallel vehicles (>= 40 eroded px, identifiable; the D34 "
            "occupancy rule left 3 vehicles and is dropped here): per-object geometry share g for vehicles whose wheel "
            "contact points are hidden vs visible; expected direction: higher when hidden (as H6).",
        "the convoy case (6.8b)": "vehicles keeping pace with the camera (0.85 <= s < 1.15): g for s < 1 and, for s >= 1, "
            "the share placed farther than the model's own single-image depth; Virtual KITTI 2 and KITTI tracking. "
            "Identifiability is not required (keeping pace means almost no parallax by construction).",
        "KITTI tracking (3.10, 6.8b)": "real vehicles with LiDAR depth when the dataset is attached: regime shares, the slope "
            "beta_geo on parallel vehicles, and the convoy case. Correspondences come from labelled boxes; the parallel "
            "threshold is the label-noise floor measured on parked vehicles.",
        "single-image protocol (6.7d)": "for the multi-view models, H11b and the CONVOY-REAL single-image slope with the image "
            "given once and twice; a model whose single-image results depend on the protocol makes no single-image claim.",
    },
    "per_object_share": "g = (e^rho - e^rho_mono) / (e^x - e^rho_mono), from the oracle's mixing model Zhat = w Zhat_mono + "
                        "(1 - w) Z_geo; computed only when |x - rho_mono| >= 0.3.",
}

# v6.3 (after reading the complete v6.2 run, before any Argoverse 2 data or any of the analyses below existed)
AMENDMENT_V6_3 = {
    "status": "written after reading the complete v6.2 run and before any Argoverse 2 data, any additional Virtual KITTI 2 "
              "pair or any of the analyses below existed. Nothing in it is a registered test; every item is exploratory "
              "and no registered verdict depends on it.",
    "exploratory_analyses": {
        "Argoverse 2 (3.11, 6.8b)": "real vehicles with LiDAR depth from the public Argoverse 2 sensor dataset (validation "
            "split, front-centre camera), downloaded automatically: regime shares, the convoy case and the slope beta_geo, "
            "with the KITTI rules of v6.2 unchanged (moving >= 0.5 m, parked <= 0.15 m, >= 40 LiDAR pixels, parallel "
            "threshold = 90th percentile of |eps| over parked vehicles, at least 1 px).",
        "matched contact comparison (6.8b)": "every Virtual KITTI 2 vehicle with hidden wheel contact is matched to the "
            "nearest vehicle with visible contact on log depth, log image area and x_geo (within 0.5 pooled SD on each, "
            "with replacement); the median paired difference of g is reported with covariate balance.",
        "more Virtual KITTI 2 pairs": "at most 150 pairs per scene instead of 60 (the short scenes stay limited by their "
            "length); same selection rule and ground truth.",
        "cluster unit (6.8b)": "intervals of the driving analyses resample vehicle tracks instead of image pairs, because "
            "the denser sampling shows the same vehicle in several pairs.",
    },
    "ground_truth_changes": "projected LiDAR (KITTI, Argoverse 2) keeps a pixel only where its return is the nearest surface "
                            "in a 5 x 5 window (no see-through between sparse returns); a moving vehicle's Argoverse 2 "
                            "returns are carried to the image time along its labelled track.",
}

# v6.4 (after reading the complete v6.3 run, before any of the data or analyses below existed)
AMENDMENT_V6_4 = {
    "status": "written after reading the complete v6.3 run and before the additional Argoverse 2 logs, the ground-contact "
              "label of any Argoverse 2 vehicle or any of the synthesis tables below existed. Nothing in it is a registered "
              "test; every item is exploratory and no registered verdict depends on it. The model outputs for the 16 Argoverse 2 "
              "logs of v6.3 existed and had been read, but never split by ground contact.",
    "exploratory_analyses": {
        "more Argoverse 2 logs (3.11b)": "48 further validation logs, taken in the same fixed random order after the logs the "
            "v6.3 stage tried, with every v6.3 rule unchanged; the two stages are pooled (64 logs) in every Argoverse 2 analysis.",
        "ground contact of real vehicles (3.11b, 6.8b)": "the four wheel contact points of every labelled Argoverse 2 vehicle "
            "as for Virtual KITTI 2 (0.10 m above the box bottom at +-0.35 length, +-0.40 width); without instance masks a "
            "point counts as visible if it lies in the image, the vehicle's own body does not hide it (the camera lies beyond "
            "the side or end face next to that wheel, and the line of sight does not cross the vehicle's own box more than "
            "0.3 m above its bottom, as a bumper overhanging the wheel would), no other labelled object's box (shrunk by 10 cm) crosses the line of "
            "sight at least 0.3 m in front of it, and no static LiDAR return in the window of columns u +- 2, rows v - 8 ... v "
            "is nearer than it by more than max(10%, 1 m). Labels, sample, contrast and matching as for Virtual KITTI 2; the "
            "image size used for matching is the projected box area times (1 - the vehicle's LiDAR occlusion). Expected "
            "direction: g higher when the contact is hidden (as H6). Check of the rule: the same rule, with the exact depth of "
            "static pixels in place of the LiDAR, is applied to the Virtual KITTI 2 vehicles and compared with their "
            "instance-mask labels (D37); agreement and Cohen's kappa are reported (T15d), with no threshold.",
        "support across testbeds (6.9)": "a descriptive synthesis of results that already exist, with no new test: the "
            "geometry share by ground support in every testbed (CONVOY-3D cars and crates against floating crates; "
            "CONVOY-REAL, whose inserted movers are all floating crates by design; the E5 and E5-REAL conditions; Virtual "
            "KITTI 2 and Argoverse 2 vehicles by contact), and the contrast of unsupported or hidden-contact objects against "
            "grounded ones in each testbed. Only the CONVOY-3D contrast (floating crates minus cars and crates, unpaired, "
            "cluster bootstrap over scenes) is new; the others are those of H5, H6, H11a and 6.8b.",
        "implied depth error (6.9)": "for each slope beta, exp(beta x) at s = 0.85 (x = -log 0.15): the depth overestimate "
            "the slope implies for an object keeping pace with the camera, with the interval mapped through the same function.",
    },
    "why": "In the v6.3 run the geometry share on CONVOY-REAL was higher than on CONVOY-3D and, for four of the six models, "
           "close to that of the CONVOY-3D floating crates; CONVOY-REAL inserts only floating crates, and the pooled CONVOY-3D "
           "slope mixes grounded and floating movers. Reporting every testbed by support was decided after seeing this pattern, which is why the "
           "synthesis is descriptive and adds no verdict.",
}

# v6.5 (after reading the complete v6.4 run, before any of the data below existed)
AMENDMENT_V6_5 = {
    "status": "written after reading the complete v6.4 run and before any of the following existed: the baseline-sweep and "
              "dose-response renders, any Bonn pair, the camera height of any Argoverse 2 vehicle. H12 and H13 are registered "
              "tests; every other item is exploratory and no registered verdict depends on it. The analyses that re-use v6.4 "
              "model outputs (the cue model, the parallax check of CONVOY-REAL, the mask trade-off) had not been run on them "
              "before this amendment, but those outputs had been read.",
    "registered_hypotheses": {
        "H12 (parallax channel: baseline sweep)": "BSW, 64 new base scenes with three purely parallel movers (cars and crates on "
            "the street, floating crates), each seen with the camera baseline scaled by 0.5, 1, 2 and 3: same first image, same "
            "rotation of the second camera, the mover's displacement scaled with the baseline, so s, eps = 0 and the geometric "
            "answer are unchanged and only the parallax changes. For DUSt3R and MASt3R: beta(x3) - beta(x0.5) > 0, paired "
            "cluster-bootstrap 95% CI > 0 (cluster = base scene), Holm across the two models, on movers measurable at every "
            "baseline and identifiable at x1. Rationale: in v6.4 these two models had beta_s of 0.55-0.61 for movers with more "
            "than 40 px of parallax and 0.07-0.17 below 40 px, and the other four showed no such rise; E1 cannot say whether "
            "parallax causes it. The same contrast is reported for the other models without a prediction.",
        "H13 (the contact junction suffices: dose-response)": "WDR, 64 new base scenes (cars and crates, purely parallel), each "
            "mover rendered without a wall and behind walls that hide at least its lowest 15, 35 and 70 cm in both views (the E5 "
            "wall, designed by ray casting; a wall must cover the whole visible base in both views, so it hides more of the "
            "mover's near end, and the share of the mover it hides is measured on the renders). For every model: beta(15 cm) - beta(no wall) > 0, CI > 0, Holm across models, on movers "
            "whose contact is visible without the wall (>= 5 contact pixels in both views) and hidden by the 15 cm wall (none). "
            "Rationale: H6 hid 70 cm of a car and 35% of a crate; the lowest wall that still hides the contact in both views "
            "tests whether hiding the contact, rather than a large part of the body, raises the geometry share.",
        "rules": "NOT TESTABLE below 8 movers or 5 base scenes; verdicts as for H5-H8 (SUPPORTED: interval > 0; CONTRADICTED: "
                 "interval < 0; NOT SUPPORTED otherwise). The static-scene gate (Section 6.0) applies to both new testbeds.",
    },
    "exploratory_analyses": {
        "shape of the dose-response and of the sweep (6.10)": "beta(70 cm) - beta(no wall), beta(70 cm) - beta(15 cm) and "
            "the share of the 70 cm effect reached at 15 cm, reported only where the 70 cm effect is clearly positive (a step "
            "points to the contact junction, a graded rise to the extent of the hidden lower body); "
            "beta by baseline for grounded and floating movers separately; M2 recall by baseline and by wall height.",
        "the static twin (Proposition 4, 3.11c, 6.13)": "on Argoverse 2 (all 64 logs) the camera height H above each vehicle's "
            "contact plane comes from its labelled box; the static twin of a vehicle then lies H (1 - e^x_geo) above that plane, "
            "below the road for a vehicle moving with the camera. Twin test: |twin height| above its 95th percentile over parked "
            "vehicles; epipolar test: |eps| above its 95th percentile over parked vehicles (5% false positives each). Recall of "
            "each test and of either, by motion regime and by s, with the detection curve the parked distribution implies. "
            "Correspondences come from labelled boxes: this measures what each test could see, not what a pipeline would.",
        "two channels: a cue model (6.11)": "per model, rho - rho_mono = w (x - rho_mono) with w = c_support d^2 / (d^2 + kappa^2) "
            "(d: translational parallax of the static-equivalent point in px; support: contact in view, contact hidden, "
            "unsupported), fitted by least squares on CONVOY-3D and E5 and compared by 10-fold cross-validation over scenes "
            "with the same model without parallax (kappa = 0) and without support (one c). Without refitting, the model then "
            "predicts the slope on BSW, WDR, CONVOY-REAL, E5-REAL and Argoverse 2; predicted and observed slopes are reported "
            "side by side. Also: binned rho against x on CONVOY-3D with the model's curve; MonST3R minus DUSt3R on the same "
            "movers (slope above 40 px of parallax, and the H6 contrast); the change of M2 recall against the change of "
            "beta_s between E5 conditions.",
        "CONVOY-REAL and parallax (6.11)": "correction of the v6.4 reading (D57): the CONVOY-REAL movers have far more parallax "
            "(median about 69 px) than the CONVOY-3D floating crates (median 13-17 px), so their higher slope may come from "
            "parallax, not from real images; the comparison is repeated within parallax bins and against the cue model.",
        "Bonn RGB-D Dynamic (3.12, 6.12)": "real balloons and boxes moved by people (four sequences), measured depth and motion-"
            "capture poses, A3 as for TUM. Moving regions as for TUM walking (Section 3.7); people masked by an off-the-shelf "
            "instance segmenter; per object: its regime (parallel threshold: the flow-noise floor of static patches, at least "
            "1 px), x_geo from the flow correspondence after a per-pair scale check on static pixels, and the height of its "
            "lowest point above the floor. All six models are evaluated on the pairs; reported are the regime shares and the "
            "per-object geometry share g of parallel objects by kind (balloon, box, person) and by floor contact. Because x_geo "
            "is built from flow and sensor depth (D24), no slope is estimated; only per-object shares with |x - rho_mono| >= "
            "0.3 enter.",
    },
    "why": "The v6.4 run showed that ground support goes with the geometry share in every testbed, and that the rise of beta_s "
           "with parallax is specific to DUSt3R and MASt3R. Two readings remained open: that parallax rather than support "
           "drives part of the effect (the CONVOY-REAL movers have far more parallax than the CONVOY-3D floating crates), and "
           "that the contact effect needs a large occluder. The two designed experiments test these with registered "
           "predictions; the twin, the cue model and Bonn connect the result to geometry and to real objects without "
           "changing any verdict.",
}

# v6.6 (after reading the complete v6.5 run, before any flow-based twin quantity or corrected Bonn pair existed)
AMENDMENT_V6_6 = {
    "status": "written after reading the complete v6.5 run and before any of the following existed: the optical flow of any "
              "Argoverse 2 pair, the LiDAR road plane of any pair, any Bonn pair with a re-estimated pose. The Argoverse 2 images, "
              "LiDAR and labels were already used in v6.5 (exact correspondences only), so H14 is registered on data that had "
              "been seen but on quantities that had not been computed. Every other item is exploratory or a correction.",
    "registered_hypotheses": {
        "H14 (the twin test survives optical flow)": "Argoverse 2, all logs, moving vehicles in the parallel regime (regime from "
            "the labelled correspondences, as in v6.5) whose flow correspondences have a static solution (at most half of "
            "their LiDAR pixels triangulate behind the camera). Both tests from the same forward-backward checked flow at the "
            "vehicle's LiDAR pixels: epipolar test, median |eps| above its 95th percentile over parked vehicles; twin test, the "
            "10th percentile of the heights of the static reconstruction above the road plane fitted by RANSAC to the LiDAR "
            "returns of the image (no labelled box) outside its 2.5-97.5% range over parked vehicles. Prediction: "
            "det_twin - det_epipolar > 0, paired cluster-bootstrap 95% CI > 0 (cluster = vehicle track). Rationale: with exact "
            "correspondences the twin test flagged parallel movers the epipolar test cannot see; a deployed system has flow, "
            "not labels, and noise in the flow could erase the twin's displacement for slow vehicles.",
        "rules": "NOT TESTABLE below 20 vehicles or 10 tracks, or below 20 parked vehicles with flow (no thresholds); verdicts "
                 "as for H5-H13. No model output enters H14.",
    },
    "corrections": {
        "Bonn ground truth (D62)": "in v6.5 the motion-capture poses (calibration and synchronisation error) made static pixels "
            "miss their rigid flow by about 2 px, so large parts of the static scene were taken for moving objects. v6.6 "
            "re-estimates the relative pose of every pair from its static correspondences (RANSAC PnP, pairs whose fit fails "
            "are left out), does not test pixels hidden in the second image, keeps a moving region only if the static "
            "prediction fails photometrically on it and if it is not larger than 20% of the image, and takes the floor as the "
            "lowest well-supported horizontal level. If the median pair still flags more than 15% of its static scene, Bonn is "
            "not interpreted. Bonn stays exploratory.",
    },
    "exploratory_analyses": {
        "contact gates parallax (6.10)": "the rise of beta_s from the x0.5 to the x3 baseline for floating minus grounded "
            "movers (the H12-interaction row that v6.5 already reported), now listed as claim C33; chosen after the v6.5 "
            "results, so descriptive.",
        "how much of the hump the cue model explains (6.11)": "the fall of the median rho from its observed peak to the bin "
            "of largest x, observed and as each fit has it.",
    },
    "why": "Two weaknesses of v6.5 were in the measurements, not in the models: the twin test used labelled correspondences, "
           "which no deployed system has, and the Bonn objects were partly static scenery. H14 asks whether the twin test "
           "still works with the correspondences a pipeline would have; the Bonn correction asks whether the real floating "
           "objects say anything once the ground truth is right.",
}

import textwrap as _tw

def _print_block(title, body):
    print("=" * 88)
    print(title)
    print("=" * 88)
    if isinstance(body, dict):
        for k, v in body.items():
            print(_tw.fill(f"{k}: {v}", width=88, subsequent_indent="    "))
    elif isinstance(body, (list, tuple)):
        for v in body:
            print(_tw.fill(f"- {v}", width=88, subsequent_indent="  "))
    else:
        print(_tw.fill(str(body), width=88))
    print()

_LOCKED_FIELDS = [
    ("LOCKED RESEARCH QUESTION", "research_question"),
    ("LOCKED HYPOTHESIS", "central_hypothesis"),
    ("LOCKED FALSIFICATION CRITERIA", "falsification_conditions"),
    ("LOCKED PROPOSITIONS", "propositions_exact_and_verified"),
    ("LOCKED ASSUMPTIONS", "assumptions"),
    ("LOCKED NUMERICAL PREDICTIONS", "falsifiable_numerical_predictions"),
    ("LOCKED BASELINES", "baselines_per_idea"),
    ("LOCKED METRICS", "main_metrics"),
    ("LOCKED CONTRIBUTION TYPE / SOTA POSITION", None),
    ("LOCKED CLAIM SCOPE", "claim_scope"),
]
for _title, _key in _LOCKED_FIELDS:
    if _key is None:
        _print_block(_title, {"contribution_type": IDEA_LOCK["contribution_type"],
                              "sota_position": IDEA_LOCK["sota_position"]})
    else:
        _print_block(_title, IDEA_LOCK[_key])
_print_block("V6 AMENDMENT: STATUS AND REFRAMING", {"status": AMENDMENT_V6["status"],
                                                   "reframing": AMENDMENT_V6["reframing"]})
_print_block("V6 AMENDMENT: REGISTERED HYPOTHESES H5-H10", AMENDMENT_V6["registered_hypotheses"])
_print_block("V6 AMENDMENT: EXPLORATORY (no prediction)", AMENDMENT_V6["exploratory_without_prediction"])
_print_block("V6 AMENDMENT: REMOVED FROM V5", AMENDMENT_V6["removed_from_v5"])
_print_block("V6.1 AMENDMENT: STATUS", AMENDMENT_V6_1["status"])
_print_block("V6.1 AMENDMENT: REGISTERED HYPOTHESES H11a-b", AMENDMENT_V6_1["registered_hypotheses"])
_print_block("V6.1 AMENDMENT: WITHDRAWN", AMENDMENT_V6_1["withdrawn"])
_print_block("V6.1 AMENDMENT: MOVED TO EXPLORATORY", AMENDMENT_V6_1["moved_to_exploratory"])
_print_block("V6.2 AMENDMENT: STATUS", AMENDMENT_V6_2["status"])
_print_block("V6.2 AMENDMENT: EXPLORATORY ANALYSES", AMENDMENT_V6_2["exploratory_analyses"])
_print_block("V6.3 AMENDMENT: STATUS", AMENDMENT_V6_3["status"])
_print_block("V6.3 AMENDMENT: EXPLORATORY ANALYSES", AMENDMENT_V6_3["exploratory_analyses"])
_print_block("V6.4 AMENDMENT: STATUS", AMENDMENT_V6_4["status"])
_print_block("V6.4 AMENDMENT: EXPLORATORY ANALYSES", AMENDMENT_V6_4["exploratory_analyses"])
_print_block("V6.4 AMENDMENT: WHY THE SYNTHESIS IS DESCRIPTIVE", AMENDMENT_V6_4["why"])
_print_block("V6.5 AMENDMENT: STATUS", AMENDMENT_V6_5["status"])
_print_block("V6.5 AMENDMENT: REGISTERED HYPOTHESES H12-H13", AMENDMENT_V6_5["registered_hypotheses"])
_print_block("V6.5 AMENDMENT: EXPLORATORY ANALYSES", AMENDMENT_V6_5["exploratory_analyses"])
_print_block("V6.5 AMENDMENT: WHY", AMENDMENT_V6_5["why"])
_print_block("V6.6 AMENDMENT: STATUS", AMENDMENT_V6_6["status"])
_print_block("V6.6 AMENDMENT: REGISTERED HYPOTHESIS H14", AMENDMENT_V6_6["registered_hypotheses"])
_print_block("V6.6 AMENDMENT: CORRECTIONS", AMENDMENT_V6_6["corrections"])
_print_block("V6.6 AMENDMENT: EXPLORATORY ANALYSES", AMENDMENT_V6_6["exploratory_analyses"])
_print_block("V6.6 AMENDMENT: WHY", AMENDMENT_V6_6["why"])
_print_block("INVENTORY (record of this notebook version, not part of the lock)",
             {"models": ", ".join(IDEA_LOCK["models_actually_registered_this_notebook"]),
              "datasets": " | ".join(IDEA_LOCK["datasets_actually_used_this_notebook"])})
print(f"{len(_LOCKED_FIELDS)} locked fields, {len(AMENDMENT_V6['registered_hypotheses'])} v6 hypotheses (H10 withdrawn and "
      f"H9 exploratory in v6.1), {len(AMENDMENT_V6_1['registered_hypotheses']) - 1} v6.1 hypotheses and "
      f"{len(AMENDMENT_V6_5['registered_hypotheses']) - 1} v6.5 hypotheses printed.")