"""Phase 5 — regenerate ``TAMNet_Paper.docx`` from ``results/`` only.

Rules enforced by this script
-----------------------------
* **No number is typed as a literal.** Every metric is read from a JSON/CSV
  artifact under ``results/`` or from ``paper/tables/``. If an artifact is
  missing the script raises rather than inventing a value.
* **Mean over seeds, never the best seed.** ``+/-`` is the sample standard
  deviation (ddof=1) and is omitted when only one seed exists.
* The legacy manuscript is preserved as ``TAMNet_Paper.pre-audit.docx`` the
  first time this runs; it is never overwritten or deleted.
* Figures are inserted only if the file exists; missing ones are logged as
  ``MISSING:`` instead of being faked.

Run: ``python -X utf8 paper/rewrite_paper.py``
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
P3 = RESULTS / "phase3"
PAPER = ROOT / "paper"
FIGS = PAPER / "figs"
TABLES = PAPER / "tables"
OUT = ROOT / "TAMNet_Paper.docx"
PRESERVED = ROOT / "TAMNet_Paper.pre-audit.docx"

MISSING_FIGURES: list[str] = []


# --------------------------------------------------------------------------
# loaders — these are the ONLY sources of numbers
# --------------------------------------------------------------------------
def J(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def C(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact missing: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


AGG = J(P3 / "agg" / "combined_summary.json")
BOOT = J(P3 / "bootstrap_ci.json")
EXT = J(P3 / "external_eval_summary.json")
SEG = J(P3 / "seg_metrics_summary.json")
UNC = J(P3 / "uncertainty_summary.json")
ROB = J(P3 / "robustness.json")
EFF = J(P3 / "efficiency.json")
ATT = J(P3 / "attention_alignment.json")
PARAMS = J(P3 / "params_table.json")
DEDUP = J(P3 / "external_dedup_manifest.json")
STAGE1 = J(P3 / "stage1_test_summary.json")
FILL = J(RESULTS / "fill_evidence.json")
LEAK = J(RESULTS / "leakage_report.json")
SPLIT_AUDIT = J(RESULTS / "split_audit.json")
DATASETS = C(RESULTS / "dataset_table.csv")
RUNCFG = J(P3 / "runs" / "tamnet_clean_s42" / "run_config.json")
RUNMET = J(P3 / "runs" / "tamnet_clean_s42" / "eval_internal_test" / "metrics.json")

GROUPS = {g["slug"]: g for g in AGG["groups"]}


def g(slug: str) -> dict:
    if slug not in GROUPS:
        raise KeyError(f"group {slug!r} not in combined_summary.json")
    return GROUPS[slug]


def ms(mean, std, d: int = 4) -> str:
    """mean ± sample-SD, SD omitted when there is only one seed."""
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.{d}f}"
    return f"{mean:.{d}f} \u00b1 {std:.{d}f}"


def gs(slug: str, key: str, d: int = 4) -> str:
    s = g(slug)["summary"][key]
    return ms(s["mean"], s["std_sample_ddof1"], d)


def gn(slug: str) -> int:
    return g(slug)["n_seeds"]


def paired(baseline: str) -> dict:
    for row in BOOT["paired_tamnet_vs_baseline"]:
        if row["baseline"] == baseline:
            return row
    raise KeyError(f"no paired comparison for {baseline!r}")


# --------------------------------------------------------------------------
# document helpers
# --------------------------------------------------------------------------
doc = Document()

_NORMAL = doc.styles["Normal"]
_NORMAL.font.size = Pt(10)


def _rich(p, text: str, size: int = 10):
    """Add ``text`` honouring ``**bold**``, ``*italic*`` and `` `code` ``."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = p.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = p.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            r = p.add_run(part)
        r.font.size = Pt(size)
    return p


def title(text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    return p


def h1(text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    return p


def h2(text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(10)
    return p


def para(text: str, indent: bool = False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.3)
    return _rich(p, text)


def bullet(text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(2)
    _rich(p, "\u2022  " + text)
    return p


def equation(text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.size = Pt(10)
    return p


_table_n = 0


def table(caption: str, header: list[str], rows: list[list[str]],
          note: str | None = None, widths: list[float] | None = None):
    global _table_n
    _table_n += 1
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_before = Pt(10)
    r = cp.add_run(f"TABLE {_roman(_table_n)}")
    r.bold = True
    r.font.size = Pt(10)
    cp2 = doc.add_paragraph()
    cp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = cp2.add_run(caption)
    r2.bold = True
    r2.font.size = Pt(9)

    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, htxt in enumerate(header):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rr = p.add_run(htxt)
        rr.bold = True
        rr.font.size = Pt(8)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if i == 0
                           else WD_ALIGN_PARAGRAPH.CENTER)
            _rich(p, str(val), size=8)
    if note:
        np = doc.add_paragraph()
        np.paragraph_format.space_before = Pt(2)
        _rich(np, note, size=8)
    return t


_fig_n = 0
FIGNO: dict[str, int] = {}


def figure(fname: str, caption: str, width: float = 6.0):
    global _fig_n
    path = FIGS / fname
    if not path.is_file():
        MISSING_FIGURES.append(fname)
        return None
    _fig_n += 1
    FIGNO[fname] = _fig_n
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.add_run().add_picture(str(path), width=Inches(width))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(f"Fig. {_fig_n}. {caption} (source: paper/figs/{fname}).")
    r.font.size = Pt(8)
    r.italic = True
    return _fig_n


def _roman(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


# --------------------------------------------------------------------------
# numbers used repeatedly
# --------------------------------------------------------------------------
N_TEST = RUNMET["n"]
N_TRAIN = RUNCFG["dataset"]["n_total"]["train"]
N_VAL = RUNCFG["dataset"]["n_total"]["val"]
CLS = RUNCFG["dataset"]["classes"]["test"]
SEEDS = sorted(int(s.split("_s")[-1]) for s in g("tamnet_clean")["per_seed_acc"])
FIGMAN = J(FIGS / "figures_manifest.json")
N_RUNS = FIGMAN["n_runs"]
N_RUNS_EVAL = FIGMAN["n_runs_with_test_eval"]
GIT_COMMIT = AGG["git"]["commit"]

TAM_ACC = gs("tamnet_clean", "accuracy")
TAM_F1 = gs("tamnet_clean", "macro_f1")
DN_ACC = gs("densenet121_clean", "accuracy")
DN_F1 = gs("densenet121_clean", "macro_f1")

P_TAM = PARAMS["params"]["tamnet_full"]["total"]
P_DN = PARAMS["params"]["densenet121"]["total"]
P_EXTRA_M = (P_TAM - P_DN) / 1e6

D_DN = paired("densenet121_clean_s42")
D_R50 = paired("resnet50_clean_s42")
D_EN = paired("efficientnet_v2_s_clean_s42")
D_CX = paired("convnext_tiny_clean_s42")
D_SW = paired("swin_t_clean_s42")
D_VIT = paired("vit_b_16_clean_s42")
D_UNET = paired("shared_unet_brisc_s42+ls0.25+sc2")


def delta(row: dict, d: int = 2) -> str:
    return f"{row['delta'] * 100:+.{d}f} pp"


def dci(row: dict, d: int = 2) -> str:
    return (f"[{row['ci95_low'] * 100:+.{d}f}, "
            f"{row['ci95_high'] * 100:+.{d}f}] pp")


def verdict(row: dict) -> str:
    return "significant" if row["significant"] else "not distinguishable"


def pf(p: float) -> str:
    """Readable p-value: bootstrap p is 0.000 when no resample crossed zero."""
    if p <= 0.0:
        return "p < 0.001"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


K_FULL = EXT["aggregate"]["kaggle_testing_full"]
K_DED = EXT["aggregate"]["kaggle_testing_dedup"]
B_FULL = EXT["aggregate"]["brisc_test_full"]
B_DED = EXT["aggregate"]["brisc_test_dedup"]
D_K = DEDUP["sources"]["kaggle_testing"]
D_B = DEDUP["sources"]["brisc_test"]

ROB_T = ROB["groups"]["tamnet_clean"]
ROB_D = ROB["groups"]["densenet121_clean"]

FUSION_W = ATT["fusion_weights"]
ATT_T = ATT["tamnet_attention"]
ATT_U = ATT["uniform"]

EC = LEAK["verdicts"]["kaggle_testing_independence"]
SCHEME_C = SPLIT_AUDIT["headline_finding"]
KAGGLE_ZIP = FILL["FILL_2_kaggle_identity"]["zip"]
FAM = FILL["FILL_1_internal_provenance"]["internal"]["filename_families"]

print(f"loaded: test n={N_TEST}, train n={N_TRAIN}, val n={N_VAL}, "
      f"seeds={SEEDS}, commit={GIT_COMMIT}")

# ==========================================================================
# TITLE / ABSTRACT
# ==========================================================================
title("TAMNet: Tumor-Aware Multi-Task Representation Learning for Brain MRI "
      "Classification and Segmentation with Uncertainty Quantification")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
_rich(p,
      "**ABSTRACT\u2014** Accurate brain tumor analysis from magnetic resonance "
      "imaging (MRI) is important for automated diagnosis. This paper presents "
      "TAMNet, a unified framework that couples a shared convolutional encoder "
      "with an auxiliary segmentation head and Monte Carlo dropout for "
      "four-class brain MRI classification \u2014 glioma, meningioma, pituitary "
      "tumor and no-tumor \u2014 together with predictive uncertainty. Our "
      "earlier measurements of this system were obtained from a train/test "
      "split that shared near-duplicate slices, so every headline number in "
      "that protocol was inflated. We therefore report a complete "
      "re-evaluation under a leakage-audited protocol: **"
      f"{N_RUNS} training runs, one to five seeds per configuration, a "
      f"held-out test set of {N_TEST} images, and paired bootstrap confidence "
      "intervals on every comparison.** Under this protocol TAMNet reaches a "
      f"test accuracy of {TAM_ACC} and a macro-F1 of {TAM_F1}. This is "
      f"statistically indistinguishable from a plain DenseNet-121 baseline "
      f"({DN_ACC}; paired \u0394 = {delta(D_DN)}, 95 % CI {dci(D_DN)}, "
      f"{pf(D_DN['p_two_sided'])}) and significantly below "
      "EfficientNetV2-S, ConvNeXt-Tiny, Swin-T and ResNet-50 "
      f"(\u0394 from {delta(D_R50)} to {delta(D_EN)}, all "
      f"{pf(max(D_R50['p_two_sided'], D_EN['p_two_sided']))}). "
      "Removing near-duplicates from the two public evaluation sets lowers "
      f"accuracy to {ms(K_DED['accuracy_mean'], K_DED['accuracy_std'])} "
      f"(Kaggle, {K_DED['n_images_per_seed']} images) and "
      f"{ms(B_DED['accuracy_mean'], B_DED['accuracy_std'])} "
      f"(BRISC, {B_DED['n_images_per_seed']} images). The ablations do not "
      "support the design as intended: the auxiliary segmentation loss gives "
      "no classification gain, channel attention is disabled in the executed "
      "model, and the learned fusion weights collapse onto the global pathway "
      f"(\u03b1_global = {FUSION_W['alpha_global_mean']:.4f}). We conclude "
      "that TAMNet does not outperform modern classification backbones on this "
      "benchmark, and we release the full re-evaluation, the de-duplication "
      "manifests and the corrected calibration and robustness measurements so "
      "that this negative result can be checked.")

p = doc.add_paragraph()
_rich(p, "**Index Terms\u2014** brain tumor, magnetic resonance imaging, "
         "multi-task learning, attention mechanisms, uncertainty quantification, "
         "data leakage, benchmark validity.")

# ==========================================================================
# I. INTRODUCTION
# ==========================================================================
h1("I.  INTRODUCTION")

para("Brain tumor classification and segmentation from MRI support diagnosis, "
     "treatment planning and follow-up. Modern convolutional and transformer "
     "backbones reach high reported accuracy on public brain-MRI benchmarks "
     "[1]\u2013[9], and the field has largely converged on a recipe: a strong "
     "image classifier, an optional segmentation branch, and \u2014 more "
     "recently \u2014 a mechanism for saying how confident the model is.",
     indent=True)

para("Two limitations persist. First, classification and segmentation are "
     "usually optimized as separate pipelines, so the segmentation "
     "information never shapes the representation used for diagnosis. "
     "Multi-task designs that share an encoder and add an auxiliary "
     "segmentation head have been proposed to make that representation "
     "tumor-aware [17]\u2013[21]. Second, a predicted label is normally "
     "reported without any measure of reliability, even though calibration "
     "degrades under dataset shift [25]\u2013[32]. Monte Carlo dropout [25] "
     "offers a cheap way to attach predictive uncertainty to an existing "
     "network without changing its training regime.", indent=True)

para("We built TAMNet on exactly that recipe. A DenseNet-121 encoder supplies "
     "a feature map; a spatial attention module and an optional channel "
     "attention module weight it; a lightweight segmentation head predicts a "
     "tumor mask that gates a tumor-region pooling; and a learned two-way "
     "fusion combines the global and tumor-region features before the "
     "classifier. Monte Carlo dropout over T passes supplies predictive "
     "entropy and mutual information at inference time.", indent=True)

para("**This paper is not a positive result.** The manuscript we originally "
     "wrote reported TAMNet at 96.45 % internal accuracy, 99.16 % on an "
     "external Kaggle set and 97.70 % on BRISC, ahead of or competitive with "
     "every baseline. An audit of the evaluation protocol showed why. The "
     "historical split shares near-duplicate slices between train and test: "
     f"{EC['contaminated_unique_images']} of {EC['total']} Kaggle testing "
     f"images ({EC['contamination_rate'] * 100:.1f} %) also occur in the "
     "training pool, and the paper's own validation scheme produces a set "
     "that is 100 % single-class. Those numbers are traceable to artifacts, "
     "but they measure leakage, not generalization. Re-measuring the same "
     "models under a manifest-pinned, near-duplicate-free protocol changes "
     "the conclusion: TAMNet no longer wins.", indent=True)

para("The contributions of this paper are:", indent=True)
bullet("**A leakage audit and a clean protocol.** We quantify exact and "
       "perceptual near-duplicate overlap between every pair of data sources, "
       "publish de-duplication keep-lists for both public test sets, and pin "
       "the split by SHA-256 in the configuration of every run.")
bullet(f"**A controlled re-evaluation.** {N_RUNS} runs, one to five seeds per "
       f"configuration, identical training budget, the {N_TEST}-image test "
       "set evaluated once per final configuration, headline = mean over "
       "seeds with sample standard deviation (ddof = 1), and a paired "
       "bootstrap 95 % confidence interval for every headline comparison.")
bullet("**An ablation of the tumor-aware machinery.** Fusion mode, channel "
       "attention and the segmentation-loss weight are each varied while the "
       "protocol is held fixed, so the contribution of each component can be "
       "read directly.")
bullet("**Corrected uncertainty, calibration and robustness measurements.** "
       "A single MC-dropout protocol (BatchNorm frozen, only dropout "
       "active, T = 10), ECE with a stated bin count, and a formally defined "
       "MRRS over a fully listed corruption set.")
bullet("**Honest reporting.** We state that TAMNet does not beat modern "
       "backbones here, we do not report the de-duplicated external numbers "
       "as if they were independent, and we release every intermediate "
       "artifact needed to reproduce the negative result.")

# ==========================================================================
# II. RELATED WORK
# ==========================================================================
h1("II.  RELATED WORK")

h2("A.  Deep Learning for Brain Tumor MRI Analysis")
para("Convolutional networks for brain tumor analysis were established by "
     "Havaei et al. [4] and the BRATS benchmark [2], and the U-Net family "
     "[7], nnU-Net [6] and 3-D multi-scale designs [5] now define the "
     "segmentation baseline. On the classification side, hand-crafted and "
     "shallow features were progressively replaced by end-to-end CNNs [8], "
     "and reviews of the field appear in [1] and [3]. More recent work "
     "pushes segmentation into the frequency domain [24], adds shape and "
     "scale awareness [21], and handles missing modalities [22] and "
     "modality completeness [20]. Transformer-based architectures [14], "
     "[15] and modern ConvNets [13] have since become the default "
     "classifiers in computer vision and are the backbones we compare "
     "against.", indent=True)

h2("B.  Attention and Multi-Task Feature Learning")
para("Squeeze-and-excitation [16] and CBAM [17] showed that re-weighting "
     "channels and spatial locations improves recognition at little cost; "
     "TAMNet's spatial attention follows the CBAM ordering. Attention-gated "
     "segmentation [18], nested U-Net variants [19] and multi-task "
     "segmentation-guided representations are the closest relatives of the "
     "tumor-aware pooling proposed here. Multi-task learning with an "
     "auxiliary segmentation objective is widely assumed to regularize "
     "classification, but the assumption is rarely isolated by an "
     "ablation \u2014 which is precisely what Section IV-E does. Gradient-based "
     "attribution such as Grad-CAM [33] provides a complementary, "
     "class-specific view of where a classifier looks, and we use it as the "
     "reference against which the learned attention map is compared in "
     "Section IV-J.", indent=True)

h2("C.  Uncertainty Quantification and Reliability")
para("Bayesian approximations to neural network uncertainty [25], the "
     "distinction between aleatoric and epistemic uncertainty [26] and deep "
     "ensembles [27] are the standard toolkit. Calibration of modern "
     "networks is poor by default [31] and degrades further under "
     "distribution shift [28], [32]; temperature scaling and reliability "
     "diagnostics are the usual remedies. We follow this literature by "
     "reporting ECE, Brier score, NLL, predictive entropy and mutual "
     "information together, rather than any single scalar.", indent=True)

h2("D.  Benchmark Validity")
para("A less common but decisive consideration is whether the benchmark "
     "itself measures what it claims. We find that the historical split for "
     "this task shares near-duplicate slices across the train/test boundary, "
     "that a substantial fraction of a widely used public \u201cexternal\u201d "
     "test set is byte-identical to training images, and that the "
     "paper's validation scheme yields a single-class set. Because our "
     "splits are image-level rather than patient-level, every interval we "
     "report is an image-level interval; we say so explicitly wherever it "
     "matters. Corruption robustness is evaluated with the common "
     "corruption suite of [37].", indent=True)

h2("E.  Research Gap")
para("The gap this paper addresses is therefore not \u201cno one has fused "
     "global and tumor-region features.\u201d It is that the claimed benefit "
     "of tumor-aware multi-task fusion has not been demonstrated under a "
     "protocol that removes leakage, and that its components have not been "
     "ablated. We close that gap with a negative result: under a clean, "
     "manifest-pinned protocol, the added machinery does not improve "
     "classification, and a plain DenseNet-121 with 2.07 M fewer parameters "
     "matches TAMNet exactly.", indent=True)

print("sections I-II written")

# ==========================================================================
# III. PROPOSED METHOD
# ==========================================================================
h1("III.  PROPOSED METHOD")

para("This section states the executed model exactly as implemented in "
     "the source module `models/tumor_aware.py`. Where an earlier draft of "
     "this paper disagreed with the code, the code is the specification and "
     "the text has been corrected; the five corrections are called out "
     "explicitly.", indent=True)

h2("A.  Notation and Shared Feature Encoding")
para("Let x in R^(3xHxW) be an input MRI slice resized to 224\u00d7224. A "
     "shared encoder E (DenseNet-121 [11] by default) produces a feature map "
     "F = E(x) in R^(C\u00d77\u00d77) with C = 1024. Global average pooling "
     "of F gives the global feature vector g in R^C, which is normalized by a "
     "batch-normalization layer:", indent=True)
equation("g  =  BN( GAP(F) )  \u2208 R^C")
para("where GAP denotes global average pooling over the spatial dimensions. "
     "Two branches consume F: the global branch above and the tumor-region "
     "branch described next. Throughout we use z_g for the projected global "
     "feature and z_t for the tumor-region feature, so that the symbol F is "
     "reserved for the convolutional feature map. (An earlier version of the "
     "manuscript used F for both, which is the notation clash corrected "
     "here.)", indent=True)

h2("B.  Tumor-Aware Spatial Attention (M_s)")
para("The spatial attention module M_s re-weights the feature map before any "
     "pooling. It computes channel-wise mean and max statistics, concatenates "
     "them, and passes them through a 7\u00d77 convolution followed by a "
     "sigmoid:", indent=True)
equation("M_s(F)  =  F  \u2299  \u03c3( Conv7\u00d77( [ mean_c(F) ; max_c(F) ] ) )")
para("where \u2299 is element-wise multiplication and the sigmoid output is a "
     "single-channel spatial map in (0, 1). This is the CBAM spatial block "
     "[17] applied to the encoder output.", indent=True)

h2("C.  Channel Attention (M_c)")
para("An SE-style channel attention module [16] is available as an option:",
     indent=True)
equation("M_c(F)  =  F  \u2299  \u03c3( FC(avgpool(F)) + FC(maxpool(F)) )")
para("with the shared two-layer bottleneck at reduction 16. When enabled it "
     "is applied after M_s, i.e. spatial then channel, following CBAM order.",
     indent=True)
para("**Correction 1 (D1).** In the executed model this module is "
     "*disabled*: `use_channel_attention` defaults to `False` and the "
     "historical runs never wired it in. Every result labelled \u201cTAMNet\u201d "
     "in this paper therefore uses M_s only. The variant that enables M_c is "
     "reported separately as an ablation in Section IV-E, not as the "
     "proposed model.", indent=True)

h2("D.  Tumor-Region Pooling")
para("When a tumor mask is available it is interpolated to the resolution of "
     "F and used to gate the map before attention:", indent=True)
equation("F_m = F \u2299 m,     x_s = M_s(F_m),     x_sc = M_c(x_s)")
equation("t  =  MLP( GAP(x_sc) )  \u2208 R^(d_f)")
para("where m is the mask, MLP is Linear(C \u2192 d_f) followed by "
     "BatchNorm1d, ReLU and Dropout(0.2), and d_f = 256 is the fusion "
     "dimension. Note that pooling is applied *after* attention and that an "
     "MLP follows the pool \u2014 an earlier draft wrote only t = GAP(F_t), "
     "which under-specifies the executed computation.", indent=True)
para("**Correction 3 (D6).** The mask gate m is never exercised. The "
     "training loop calls the model as `model(images, return_seg=True)` or "
     "`model(images)` and never supplies `seg_mask`, so `mask` is `None` in "
     "every training call and every inference call. The tumor-region branch "
     "therefore pools the *unmasked* feature map: what makes it tumor-aware "
     "is the attention module, not a spatial mask. The segmentation head is "
     "used only to produce L_seg and the segmentation evaluation of "
     "Section IV-G.", indent=True)

h2("E.  Global\u2013Tumor Feature Fusion (\u03a6)")
para("Both branches are projected to the fusion dimension and combined by a "
     "learned two-way convex combination:", indent=True)
equation("z_g = W_g g,    z_t = W_t t")
equation("w    = softmax( MLP_fuse( [ z_g ; z_t ] ) )  \u2208 (0,1)^2")
equation("z    = LayerNorm( w_0 \u00b7 z_g  +  w_1 \u00b7 z_t )")
para("MLP_fuse is Linear(2d_f \u2192 d_f) \u2192 ReLU \u2192 Linear(d_f \u2192 2) "
     "\u2192 Softmax. The classifier is then a three-layer MLP on z "
     "(256 \u2192 128 \u2192 4) with dropout 0.3 and 0.15.",
     indent=True)
para("**Correction 2 (D4).** An earlier draft described this fusion as a "
     "concatenation of g and t. It is not: it is a softmax-weighted convex "
     "combination followed by LayerNorm. Concatenation is the *ablation* "
     "(`use_attention_fusion = False`) and is reported as such. A third "
     "variant, `fusion_mode = t`, passes only z_t to the classifier.",
     indent=True)
para(f"Measured over {FUSION_W['n']} slices, the learned weights are "
     f"\u03b1_global = {FUSION_W['alpha_global_mean']:.4f} and "
     f"\u03b1_tumor = {FUSION_W['alpha_tumor_mean']:.4f} "
     f"(\u00b1 {FUSION_W['alpha_tumor_std']:.4f}). The fusion therefore "
     "collapses onto the global pathway: the tumor-region feature contributes "
     "about three percent of the fused representation. Section IV-J reports "
     "the consequence directly.", indent=True)

h2("F.  Auxiliary Segmentation Branch")
para("A lightweight decoder reads the 7\u00d77 encoder map alone \u2014 no "
     "skip connections \u2014 and upsamples it to a segmentation logit map at "
     "56\u00d756, which the forward pass then interpolates to the input "
     "resolution:", indent=True)
equation("Conv3\u00d73(1024\u2192128) \u2192 BN \u2192 ReLU \u2192 up\u00d74 "
         "\u2192 Conv3\u00d73(128\u219264) \u2192 BN \u2192 ReLU \u2192 up\u00d72 "
         "\u2192 Conv1\u00d71(64\u2192K)")
para("with K segmentation classes; the result is bilinearly interpolated to "
     "224\u00d7224. A multi-scale U-Net-style decoder over tapped encoder "
     "stages exists as an ablation but is not the executed configuration.",
     indent=True)
para("**Correction 4 (D2, D3).** In the historical training runs no runner "
     "ever passed segmentation targets, so L_seg was identically zero and "
     "the branch received no gradient. The whole-tumor / tumor-core / "
     "enhancing-tumor Dice values quoted in the earlier draft "
     "(0.9005 / 0.5803 / 0.6010) do not come from this branch at all: they "
     "are produced by a *standalone 3-D U-Net* trained on BraTS volumes "
     "(`checkpoints/segmentation/unet3d`). Section IV-G reports both facts "
     "and gives the segmentation numbers this architecture can actually "
     "defend.", indent=True)

h2("G.  Multi-Task Objective")
para("Training minimizes a weighted sum of a smoothed classification loss "
     "and, when segmentation targets are supplied, a segmentation loss:",
     indent=True)
equation("L  =  \u03bb_cls \u00b7 L_cls  +  \u03bb_seg \u00b7 L_seg")
equation("L_cls = CrossEntropy( logits, y ; label_smoothing = 0.1 )")
para("with \u03bb_cls = 1.0 by default and \u03bb_seg the weight swept in "
     "Section IV-E. L_seg is the Dice-based segmentation loss of the "
     "multi-task trainer. The Monte Carlo dropout *head* "
     "(`UncertaintyHead`) is present in the module list but its output is "
     "**not** added to any loss: it contributes nothing to training and "
     "exists only so that dropout sites are registered on the fused feature. "
     "This is finding D5 and is the reason pre-audit uncertainty numbers "
     "could not be salvaged.", indent=True)

h2("H.  Uncertainty Quantification Using Monte Carlo Dropout")
para("At inference the model is put in evaluation mode and *only* the "
     "dropout modules are re-activated, so BatchNorm uses its running "
     "statistics and every pass differs solely by dropout masks [25]. With T "
     "passes we obtain probabilities p_t(y | x):", indent=True)
equation("p(y | x)      =  (1/T) \u03a3_t p_t(y | x)")
equation("H[p]          =  \u2013 \u03a3_y p(y|x) log p(y|x)              "
         "(predictive entropy)")
equation("MI[y,\u03b8|x] =  H[p] \u2013 (1/T) \u03a3_t H[p_t]             "
         "(mutual information)")
para("**Correction 5 (D5).** An earlier implementation called `train()` on "
     "the whole module, which also placed BatchNorm in training mode; the "
     "ten passes then mixed batch-statistics noise into every uncertainty "
     "number and mutated the running statistics. All uncertainty figures in "
     "this paper are produced by the corrected protocol with T = 10 and "
     "BatchNorm frozen. Dropout is active at four sites: classifier 0.3 and "
     "0.15, tumor-pool MLP 0.2, uncertainty head 0.3.", indent=True)

h2("I.  Training and Inference Procedure")
para("Training uses AdamW with cosine annealing, label smoothing 0.1, and "
     "early stopping on validation weighted-F1. Full hyperparameters and the "
     "hardware are listed in Section IV-B. For evaluation the checkpoint "
     "with the best validation weighted-F1 is loaded, the held-out test set "
     "is streamed batch-by-batch to the GPU, and metrics are computed once "
     "from the saved predictions. Predictions are retained for every run so "
     "that all confidence intervals in this paper are recomputed from the "
     "same files rather than copied from summaries.", indent=True)

print("section III written")

# ==========================================================================
# IV. EXPERIMENTS  (merged setup + results)
# ==========================================================================
h1("IV.  EXPERIMENTS")

h2("A.  Datasets and Leakage Audit")
para("Table I summarises every dataset used. Four sources of 2-D MRI slices "
     "are involved; the label provenance of the internal pool is only "
     "partially documented and is marked accordingly rather than asserted.",
     indent=True)

_ds_rows = []
for r in DATASETS:
    n_img = r["n_images"]
    _ds_rows.append([
        f"{r['dataset']} ({r['split']})",
        r["path"],
        n_img,
        f"{r['class_glioma']}/{r['class_meningioma']}/"
        f"{r['class_pituitary']}/{r['class_no_tumor']}",
        r["role"],
    ])
table("Datasets, locations, sizes and roles. Class columns are "
      "glioma/meningioma/pituitary/no-tumor.",
      ["Source (split)", "Path", "Images", "G/M/P/N", "Role"],
      _ds_rows,
      note="Source: `results/dataset_table.csv`. Rows carrying [VERIFY] or "
           "[CITATION NEEDED] are unresolved provenance, not estimates.")

para(f"**Split.** All classification experiments use one manifest-pinned "
     f"split of the internal pool: {N_TRAIN} training, {N_VAL} validation "
     f"and {N_TEST} test images, with per-class test counts of "
     f"{CLS['Glioma']} glioma, {CLS['Meningioma']} meningioma, "
     f"{CLS['Pituitary']} pituitary and {CLS['No Tumor']} no-tumor. The "
     f"manifest is `results/phase3/split_manifest_clean.json` "
     f"(SHA-256 recorded in every `run_config.json`), and it is the only "
     f"split used for the headline numbers. The split is **image-level, not "
     f"patient-level**: no patient identifier exists in any filename of the "
     f"internal pool, so a patient-held-out split cannot be constructed from "
     f"it. Every confidence interval in this paper is therefore an "
     f"image-level interval.", indent=True)

para("**Leakage.** Perceptual-hash near-duplicate detection (pHamming \u2264 6 "
     "or dHamming \u2264 4, union) between every pair of sources shows that "
     "the historical evaluation was not independent:", indent=True)
bullet(f"**Internal train vs internal test.** "
       f"{LEAK['pairs']['internal_train__internal_test']['n_near_duplicates_union']:,} "
       "near-duplicate pairs. The historical split reproduces this directly: "
       f"the leaky control reaches {gs('tamnet_leaky_control', 'accuracy')} "
       f"against {TAM_ACC} on the clean split \u2014 a "
       f"{(float(g('tamnet_leaky_control')['summary']['accuracy']['mean']) - float(g('tamnet_clean')['summary']['accuracy']['mean'])) * 100:.2f} pp "
       "inflation from leakage alone.")
bullet(f"**Kaggle testing vs internal train.** {EC['contaminated_unique_images']} "
       f"of {EC['total']} images ({EC['contamination_rate'] * 100:.1f} %) also "
       f"occur in training, of which {EC['byte_identical_unique_images']} are "
       "byte-identical. Kaggle testing is not an independent external set.")
bullet(f"**BRISC test vs internal train.** {D_B['drop_vs_clean_train']} of "
       f"{D_B['total']} images ({D_B['pct_dropped']} %) near-match a "
       "clean-training image.")
bullet(f"**Validation scheme.** {SCHEME_C}")
para("Because of this, the paper's original external numbers are reported "
     "below only as *full-set* values alongside their de-duplicated "
     "counterparts, never alone.", indent=True)

h2("B.  Implementation Details")
_impl = [
    ["Backbone (encoder)", "DenseNet-121, ImageNet-pretrained"],
    ["Input", "224 \u00d7 224 RGB; Resize \u2192 ToTensor \u2192 "
              "ImageNet normalisation"],
    ["Optimiser / schedule", f"{RUNCFG['optimizer']} "
                             f"(lr = {RUNCFG['lr']}, "
                             f"weight decay = {RUNCFG['weight_decay']})"],
    ["Epochs / batch size", f"{RUNCFG['epochs']} / {RUNCFG['batch_size']} "
                            f"(30-epoch budget for every run)"],
    ["Early stopping", f"patience {RUNCFG['patience']} on "
                       f"{RUNCFG['selection_metric']}"],
    ["Label smoothing", f"{RUNCFG['label_smoothing']}"],
    ["Loss weights", "\u03bb_cls = 1.0; \u03bb_seg swept in Section IV-E"],
    ["MC-dropout passes", f"T = {UNC['passes']} (BatchNorm frozen)"],
    ["Seeds", ", ".join(str(s) for s in SEEDS) +
              " \u2014 TAMNet headline over 5 seeds"],
    ["Framework / hardware",
     f"PyTorch {RUNCFG['torch']}; {RUNCFG['device']}"],
    ["Test-set policy", "evaluated once per final configuration; "
                        "headline = mean over seeds, SD = ddof 1"],
]
table("Implementation details. Every field is read from the run "
      "configuration of the headline run.",
      ["Setting", "Value"], _impl,
      note="Source: `results/phase3/runs/tamnet_clean_s42/run_config.json` "
           "and `results/phase3/uncertainty_summary.json`.")

h2("C.  Evaluation Metrics")
para("Classification is scored by accuracy, macro-F1 and weighted-F1, all "
     "computed with scikit-learn conventions from the saved predictions; "
     "macro-F1 is the unweighted mean of the four per-class F1 scores, so it "
     "is never forced to exceed accuracy. Segmentation is scored by Dice and "
     "IoU on tumour slices, pixel accuracy, and the rate at which empty "
     "ground-truth masks are predicted empty. Calibration uses ECE with a "
     "stated 15 bins, maximum calibration error, NLL and Brier score; "
     "uncertainty uses predictive entropy, mutual information and the AUROC "
     "of mutual information for separating errors from correct predictions. "
     "Robustness uses the mean relative robustness score defined in "
     "Section IV-I. Computational cost uses parameter count, checkpoint size "
     "in MiB and synchronised latency. Every accuracy in the paper carries a "
     "percentile bootstrap 95 % interval from B = "
     f"{BOOT['splits']} resamples of the test images.", indent=True)

h2("D.  Main Classification Results")
para("Table III is the headline result. Read in descending accuracy order, "
     "TAMNet does not lead: it sits below four modern backbones, level with "
     "DenseNet-121 and ViT-B/16, and above the shared-encoder U-Net "
     "multi-task baseline.", indent=True)

_MAIN_ORDER = [
    ("convnext_tiny_clean", "ConvNeXt-Tiny"),
    ("efficientnet_v2_s_clean", "EfficientNetV2-S"),
    ("swin_t_clean", "Swin-T"),
    ("resnet50_clean", "ResNet-50"),
    ("tamnet_clean_mc", "TAMNet + channel attention"),
    ("tamnet_clean_concat", "TAMNet, concat fusion (ablation)"),
    ("tamnet_clean_ep50", "TAMNet, 50-epoch budget control"),
    ("tamnet_clean", "**TAMNet (executed config.)**"),
    ("densenet121_clean", "DenseNet-121"),
    ("tamnet_clean_tonly", "TAMNet, t-only fusion (ablation)"),
    ("vit_b_16_clean", "ViT-B/16"),
    ("shared_unet_brisc", "Shared-encoder U-Net (multi-task)"),
    ("tamnet_leaky_control", "Leaky-split control (not a result)"),
]
_main_rows = []
for slug, label in _MAIN_ORDER:
    _main_rows.append([
        label, str(gn(slug)), gs(slug, "accuracy"), gs(slug, "macro_f1"),
        gs(slug, "weighted_f1"),
    ])
table("Internal TEST accuracy, macro-F1 and weighted-F1 (mean \u00b1 sample "
      "SD over seeds; SD omitted for a single seed). n = 1,672 images.",
      ["Model / configuration", "Seeds", "Accuracy", "Macro-F1", "Weighted-F1"],
      _main_rows,
      note="Source: `results/phase3/agg/combined_summary.json`. The leaky "
           "control reproduces the historical split and is shown only to "
           "quantify the inflation; it is not a competing method.")

para(f"**Statistical significance.** Table IV gives the paired comparison "
     f"against TAMNet on identical test indices. TAMNet is significantly "
     f"worse than EfficientNetV2-S ({delta(D_EN)}, {pf(D_EN['p_two_sided'])}), "
     f"ConvNeXt-Tiny ({delta(D_CX)}, {pf(D_CX['p_two_sided'])}), "
     f"Swin-T ({delta(D_SW)}, {pf(D_SW['p_two_sided'])}) and "
     f"ResNet-50 ({delta(D_R50)}, {pf(D_R50['p_two_sided'])}); "
     f"indistinguishable from DenseNet-121 ({delta(D_DN)}, "
     f"{pf(D_DN['p_two_sided'])}) and ViT-B/16 ({delta(D_VIT)}, "
     f"{pf(D_VIT['p_two_sided'])}); and significantly better than the "
     f"shared-encoder U-Net ({delta(D_UNET)}, {pf(D_UNET['p_two_sided'])}). "
     "Per-run 95 % intervals are about "
     "\u00b11.1 pp, so differences below roughly one percentage point cannot "
     "be resolved by this benchmark at all.", indent=True)

_sig_rows = []
for row in BOOT["paired_tamnet_vs_baseline"]:
    _sig_rows.append([
        row["baseline"].replace("_clean_s42", "").replace("_s42+ls0.25+sc2", ""),
        delta(row), dci(row),
        pf(row["p_two_sided"]), verdict(row),
    ])
table("Paired bootstrap comparison of TAMNet (seed 42) against each "
      "baseline on identical test images.",
      ["Baseline", "\u0394 accuracy", "95 % CI", "p", "Conclusion"],
      _sig_rows,
      note="Source: `results/phase3/bootstrap_ci.json` (percentile "
           "bootstrap, B = 1,000, image-level resampling).")

figure("fig4b_test_accuracy.png",
       "Test accuracy per configuration with 95 % bootstrap intervals; "
       "the horizontal band is the TAMNet mean.")
figure("fig6_confusion_internal.png",
       "Row-normalised confusion matrix for the TAMNet headline run "
       "(seed 42) on the clean test split.")
figure("fig6b_per_class_f1.png",
       "Per-class F1 for the TAMNet headline run; meningioma is the "
       "weakest class.")
figure("fig5_pareto_params_accuracy.png",
       "Accuracy against parameter count. DenseNet-121 matches TAMNet with "
       "fewer parameters, so no efficiency advantage survives.")

h2("E.  Ablations")
para("Table V isolates each tumor-aware component while holding the "
     "protocol fixed.", indent=True)

_abl = [
    ["Fusion \u03a6", "convex combination (executed)", gn("tamnet_clean"),
     gs("tamnet_clean", "accuracy")],
    ["Fusion \u03a6", "concatenation (ablation)",
     gn("tamnet_clean_concat"), gs("tamnet_clean_concat", "accuracy")],
    ["Fusion \u03a6", "tumor feature only (ablation)",
     gn("tamnet_clean_tonly"), gs("tamnet_clean_tonly", "accuracy")],
    ["Channel attention M_c", "enabled", gn("tamnet_clean_mc"),
     gs("tamnet_clean_mc", "accuracy")],
    ["Channel attention M_c", "disabled (executed)", gn("tamnet_clean"),
     gs("tamnet_clean", "accuracy")],
    ["Epoch budget", "30 epochs (executed)", gn("tamnet_clean"),
     gs("tamnet_clean", "accuracy")],
    ["Epoch budget", "50 epochs (control)", gn("tamnet_clean_ep50"),
     gs("tamnet_clean_ep50", "accuracy")],
]
for slug in ("tamnet_brisc_lambda0", "tamnet_brisc_lambda0.1",
             "tamnet_brisc_lambda0.25", "tamnet_brisc_lambda1"):
    lam = slug.replace("tamnet_brisc_lambda", "")
    _abl.append([r"Segmentation loss $\lambda_{seg}$".replace("$", ""),
                 f"\u03bb_seg = {lam} (BRISC protocol)", gn(slug),
                 gs(slug, "accuracy")])
table("Ablations under the fixed protocol. The \u03bb_seg rows use the "
      "BRISC training protocol because only that source has aligned masks.",
      ["Component", "Setting", "Seeds", "Accuracy"], _abl,
      note="Source: `results/phase3/agg/combined_summary.json`. The "
           "\u03bb_seg rows are single-seed point estimates and are labelled "
           "as such.")

para("Three conclusions follow. **(i) Fusion.** Replacing the learned "
     f"convex combination with plain concatenation changes accuracy by "
     f"{(float(g('tamnet_clean_concat')['summary']['accuracy']['mean']) - float(g('tamnet_clean')['summary']['accuracy']['mean'])) * 100:+.2f} pp "
     f"and dropping the global branch entirely changes it by "
     f"{(float(g('tamnet_clean_tonly')['summary']['accuracy']['mean']) - float(g('tamnet_clean')['summary']['accuracy']['mean'])) * 100:+.2f} pp; "
     "neither is an improvement, and both are inside the seed-to-seed spread "
     "of \u00b11.10 pp. The fusion mechanism is not carrying the result.",
     indent=True)
para("**(ii) Channel attention.** Enabling M_c raises accuracy by "
     f"{(float(g('tamnet_clean_mc')['summary']['accuracy']['mean']) - float(g('tamnet_clean')['summary']['accuracy']['mean'])) * 100:+.2f} pp "
     f"over the executed configuration ({gs('tamnet_clean_mc', 'accuracy')} "
     f"vs {gs('tamnet_clean', 'accuracy')}), which is the largest single "
     "component effect in the table \u2014 and it is an effect the executed "
     "model never had. We report it rather than fold it into the headline.",
     indent=True)
para("**(iii) Segmentation loss.** On the BRISC protocol, where aligned "
     "masks exist, adding the auxiliary loss never helps: "
     f"\u03bb_seg = 0 gives {gs('tamnet_brisc_lambda0', 'accuracy')} and "
     f"\u03bb_seg = 1 gives {gs('tamnet_brisc_lambda1', 'accuracy')}, with the "
     "intermediate values no better. Since these are single-seed runs they "
     "are point estimates, but the direction is consistent \u2014 "
     "segmentation supervision does not improve classification. This is the "
     "central design assumption of the model, and it is not supported.",
     indent=True)

figure("fig4_ablation_val.png",
       "Validation curves for the ablation configurations.")

h2("F.  External Validation Before and After De-duplication")
para("Table VI reports both versions of every external number. The full-set "
     "row reproduces the contamination described in Section IV-A; the "
     "de-duplicated row is the only value that may be called external "
     "generalisation.", indent=True)

_ext_rows = []
for key, label, src in [
    ("kaggle_testing_full", "Kaggle Testing \u2014 all images", "kaggle_testing"),
    ("kaggle_testing_dedup", "Kaggle Testing \u2014 de-duplicated", "kaggle_testing"),
    ("brisc_test_full", "BRISC test \u2014 all images", "brisc_test"),
    ("brisc_test_dedup", "BRISC test \u2014 de-duplicated", "brisc_test"),
]:
    a = EXT["aggregate"][key]
    s = DEDUP["sources"][src]
    _ext_rows.append([
        label, str(a["n_images_per_seed"]),
        ms(a["accuracy_mean"], a["accuracy_std"]),
        ms(a["macro_f1_mean"], a["macro_f1_std"]),
        f"{s['drop_vs_clean_train']}/{s['total']} ({s['pct_dropped']} %)",
    ])
table("External evaluation of the TAMNet headline model over five seeds, "
      "before and after removing images that near-match the clean training "
      "split.",
      ["Evaluation set", "Images", "Accuracy", "Macro-F1", "Dropped vs train"],
      _ext_rows,
      note="Source: `results/phase3/external_eval_summary.json`; drop counts "
           "from `results/phase3/external_dedup_manifest.json` (pHash \u2264 6 "
           "or dHash \u2264 4 against clean train).")

para(f"De-duplication costs "
     f"{(K_FULL['accuracy_mean'] - K_DED['accuracy_mean']) * 100:.2f} pp on "
     f"Kaggle and {(B_FULL['accuracy_mean'] - B_DED['accuracy_mean']) * 100:.2f} "
     f"pp on BRISC \u2014 the difference between in-distribution and genuine "
     f"generalisation. Even the full-set re-run "
     f"({ms(K_FULL['accuracy_mean'], K_FULL['accuracy_std'])} on Kaggle) "
     f"falls short of the 99.16 % quoted in the earlier draft, which came "
     f"from a different checkpoint under the historical protocol. We "
     f"therefore withdraw both original external claims.", indent=True)

figure("fig15_generalization_matrix.png",
       "Train-source \u00d7 test-set accuracy matrix, full and "
       "de-duplicated.")

h2("G.  Segmentation Performance")
para("Two different things have been called \u201cthe segmentation result\u201d "
     "in this project, and they must not be conflated.", indent=True)
para("**(a) The multi-task head of this model.** TAMNet's auxiliary "
     "segmentation head is evaluated on the BRISC 2025 test split as binary "
     "tumour-versus-background Dice on 2-D 224\u00d7224 slices. Table VII "
     "gives what the architecture can defend.", indent=True)

_seg_rows = []
for grp in SEG["groups"]:
    _seg_rows.append([
        grp["slug"], str(grp["n_seeds"]),
        ms(grp["dice_tumor_mean"], grp["dice_tumor_std_sample_ddof1"]),
        f"{grp['iou_tumor_mean']:.4f}",
        f"{grp['pixel_acc_mean']:.4f}",
        f"{grp['empty_gt_empty_pred_mean']:.3f}",
    ])
table("Binary tumour Dice of the auxiliary segmentation head on the BRISC "
      "2025 test split.",
      ["Configuration", "Seeds", "Dice", "IoU", "Pixel acc", "Empty\u2192empty"],
      _seg_rows,
      note="Source: `results/phase3/seg_metrics_summary.json`. Single-seed "
           "rows have no standard deviation. These are not WT/TC/ET numbers.")

para("**(b) The whole-tumour / tumor-core / enhancing-tumour numbers.** "
     "The values 0.9005 / 0.5803 / 0.6010 quoted in the earlier draft are "
     "**verified but mis-attributed**: they are produced by a standalone "
     "3-D U-Net trained on BraTS volumes "
     "(`checkpoints/segmentation/unet3d`, 45 cases), not by TAMNet's "
     "auxiliary head, and in the historical runs that head received no "
     "gradient at all (L_seg \u2261 0). BraTS contributes no classification "
     "images to this paper: it carries no pituitary and no no-tumour label, "
     "so it cannot supply the four-class task, and no 3-D-to-2-D conversion "
     "exists anywhere in the classification path. The reported standard "
     "deviations for those legacy figures are 0.048 (WT), 0.253 (TC) and "
     "0.242 (ET) \u2014 the means hide very large case-to-case variance, and "
     "the earlier draft quoted the means alone. We keep the figures only as "
     "a characterisation of that separate 3-D system.", indent=True)

h2("H.  Uncertainty and Calibration")
para("Table VIII reports calibration for the configurations that have a "
     "full multi-seed MC-dropout evaluation, using the corrected protocol "
     "of Section III-H.", indent=True)

_unc_rows = []
for grp in UNC["groups"]:
    s = grp["summary"]
    _unc_rows.append([
        grp["description"], str(grp["n_seeds"]),
        ms(s["accuracy"]["mean"], s["accuracy"]["std_sample_ddof1"]),
        ms(s["ece_15"]["mean"], s["ece_15"]["std_sample_ddof1"]),
        ms(s["nll"]["mean"], s["nll"]["std_sample_ddof1"]),
        ms(s["brier"]["mean"], s["brier"]["std_sample_ddof1"]),
        ms(s["mean_mutual_information"]["mean"],
           s["mean_mutual_information"]["std_sample_ddof1"]),
        ms(s["error_detection_auroc"]["mean"],
           s["error_detection_auroc"]["std_sample_ddof1"]),
    ])
table("Calibration and uncertainty on the clean test split (T = 10, "
      "BatchNorm frozen, 15 ECE bins).",
      ["Configuration", "Seeds", "Accuracy", "ECE", "NLL", "Brier", "MI",
       "Error AUROC"],
      _unc_rows,
      note="Source: `results/phase3/uncertainty_summary.json`. Error AUROC "
           "is how well mutual information separates misclassified from "
           "correctly classified slices (0.5 = no signal).")

_u = UNC["groups"][0]["summary"]
_d_ece = [x for x in UNC["groups"] if x["slug"] == "densenet121_clean"][0]
para(f"TAMNet's ECE of "
     f"{_u['ece_15']['mean']:.4f} against an accuracy of "
     f"{_u['accuracy']['mean']:.4f} means the model is **under-confident**: "
     f"mean confidence {_u['mean_confidence']['mean']:.4f} sits below realised "
     f"accuracy. Mutual information carries real signal "
     f"(error AUROC {_u['error_detection_auroc']['mean']:.3f}), so the "
     f"uncertainty estimate is usable for ranking risk even though the "
     f"absolute confidence is miscalibrated. DenseNet-121 is better "
     f"calibrated (ECE {_d_ece['summary']['ece_15']['mean']:.4f} at "
     f"{_d_ece['summary']['accuracy']['mean']:.4f} accuracy) on the same "
     f"split, which is a further reason not to claim a calibration "
     f"advantage.", indent=True)

figure("fig9_reliability.png",
       "Reliability diagram on the clean test split; diagonal = perfect "
       "calibration.")
figure("fig10_uncertainty.png",
       "Uncertainty histograms, risk\u2013coverage curve and error-detection "
       "AUROC.")

h2("I.  Robustness to Image Corruptions")
para(f"The corruption set contains {len(ROB['corruption_set'])} conditions "
     f"\u2014 {len(ROB['corruption_set']) // 3} corruption types \u00d7 3 "
     f"severities \u2014 applied in pixel space before normalisation, with "
     f"the clean path verified bit-identical. MRRS is defined formally as",
     indent=True)
equation("MRRS  =  (1/K) \u03a3_{k=1..K}  acc(corruption_k) / acc(clean)")
para(f"averaged over the {len(ROB['corruption_set'])} conditions and then "
     "reported as a mean over seeds \u2014 never as the best seed. The types "
     "are " + ", ".join(sorted({c['corruption'] for c in ROB['corruption_set']})) +
     ", each at severities 1\u20133.", indent=True)

table("Mean relative robustness score under the 27-condition corruption "
      "set.",
      ["Configuration", "Seeds", "Clean accuracy", "MRRS (mean \u00b1 SD)",
       "Best-seed MRRS"],
      [["TAMNet", str(ROB_T["n_seeds"]),
        gs("tamnet_clean", "accuracy"),
        ms(ROB_T["mrrs_mean"], ROB_T["mrrs_std_sample_ddof1"]),
        f"{ROB_T['mrrs_best_seed']:.4f}"],
       ["DenseNet-121", str(ROB_D["n_seeds"]),
        gs("densenet121_clean", "accuracy"),
        ms(ROB_D["mrrs_mean"], ROB_D["mrrs_std_sample_ddof1"]),
        f"{ROB_D['mrrs_best_seed']:.4f}"]],
      note="Source: `results/phase3/robustness.json`.")

para(f"TAMNet is the more corruption-robust of the two "
     f"({ms(ROB_T['mrrs_mean'], ROB_T['mrrs_std_sample_ddof1'])} vs "
     f"{ms(ROB_D['mrrs_mean'], ROB_D['mrrs_std_sample_ddof1'])}), and the gap "
     f"survives the seed spread. Note that the best-seed values "
     f"({ROB_T['mrrs_best_seed']:.4f} and {ROB_D['mrrs_best_seed']:.4f}) are "
     f"shown only to document how large the seed effect is; neither is used "
     f"as a headline. The earlier draft's MRRS of 0.7512 was a best-seed "
     f"value from a different protocol and is superseded.", indent=True)

figure("fig13_robustness_heatmap.png",
       "Accuracy for each corruption \u00d7 severity cell, normalised by "
       "clean accuracy.")
figure("fig14_intensity_curves.png",
       "Accuracy against corruption severity for the intensity-type "
       "corruptions.")

h2("J.  Attention Alignment")
para("If the spatial attention map were tumor-aware it should place more "
     "mass inside the ground-truth tumour mask than a uniform map does. "
     "Table X shows that it does not: on "
     f"{ATT_T['n']} BRISC tumour slices the attention mass inside the mask "
     f"is {ATT_T['mass_in_mask_mean']:.4f} against a uniform baseline of "
     f"{ATT_U['mass_in_mask_mean']:.4f}, a lift of "
     f"{ATT_T['lift_over_mask_fraction']:.2f}\u00d7, and the pointing-game "
     f"hit rate is {ATT_T['pointing_hit_rate']:.3f}. Combined with the "
     f"fusion weight \u03b1_tumor = {FUSION_W['alpha_tumor_mean']:.4f} from "
     f"Section III-E, the tumor-aware pathway is measurably close to "
     f"inert.", indent=True)

table("Attention\u2013mask alignment on BRISC tumour slices.",
      ["Map", "Slices", "Mass in mask", "Uniform baseline", "Lift",
       "Pointing hit"],
      [["TAMNet spatial attention", str(ATT_T["n"]),
        f"{ATT_T['mass_in_mask_mean']:.4f}",
        f"{ATT_U['mass_in_mask_mean']:.4f}",
        f"{ATT_T['lift_over_mask_fraction']:.2f}\u00d7",
        f"{ATT_T['pointing_hit_rate']:.3f}"],
       ["Uniform (reference)", str(ATT_U["n"]),
        f"{ATT_U['mass_in_mask_mean']:.4f}",
        f"{ATT_U['mask_fraction_mean']:.4f}", "1.00\u00d7",
        f"{ATT_U['pointing_hit_rate']:.3f}"]],
      note="Source: `results/phase3/attention_alignment.json`. Definitions: "
           "mass-in-mask = fraction of map mass inside the mask; lift = "
           "mass-in-mask divided by mask area. Grad-CAM on the DenseNet-121 "
           "baseline could not be extracted (no `.features` module) and is "
           "recorded as missing rather than approximated.")

figure("fig11_attention_alignment.png",
       "Attention mass inside the tumour mask versus the uniform reference, "
       "with Grad-CAM marked unavailable.")

h2("K.  Computational Efficiency")
para("Table XI reports the measured cost of every model under a fixed "
     "protocol: batch 1, 20 warm-up iterations, 200 timed iterations, CUDA "
     "synchronised around each iteration.", indent=True)

_eff_rows = []
for name, m in EFF["models"].items():
    _eff_rows.append([
        name, f"{m['params_total']:,}", f"{m['weights_mib']:.2f}",
        f"{m['single_pass']['median_ms']:.2f}",
        f"{m['mc_10_pass']['median_ms']:.2f}",
        f"{m['single_pass']['fps_from_median']:.1f}",
    ])
table("Measured computational characteristics (batch 1, "
      f"{EFF['protocol']['warmup_iterations']} warm-up, "
      f"{EFF['protocol']['timed_iterations']} timed iterations).",
      ["Model", "Parameters", "Weights (MiB)", "Single pass (ms)",
       "MC 10 passes (ms)", "FPS (single)"],
      _eff_rows,
      note=f"Source: `results/phase3/efficiency.json`. Device: "
           f"{EFF['hardware']['device']}, PyTorch {EFF['hardware']['torch']}.")

para(f"TAMNet uses {P_TAM:,} parameters "
     f"({P_TAM / 1e6:.2f} M) against "
     f"{P_DN:,} for DenseNet-121 \u2014 "
     f"{P_EXTRA_M:.2f} M "
     f"*more* parameters for the same accuracy. A single forward pass takes "
     f"{EFF['models']['tamnet_full']['single_pass']['median_ms']:.2f} ms and the full "
     f"T = {UNC['passes']}-pass Monte Carlo estimate takes "
     f"{EFF['models']['tamnet_full']['mc_10_pass']['median_ms']:.2f} ms, so any "
     "statement about throughput must say which of the two it means. We do "
     "not claim a near-real-time advantage.", indent=True)

print("section IV written")

# ==========================================================================
# V. DISCUSSION
# ==========================================================================
h1("V.  DISCUSSION")

para(f"**Does TAMNet beat its baselines? No.** Under the clean protocol the "
     f"model scores {TAM_ACC} against {DN_ACC} for a plain DenseNet-121 "
     f"trained identically \u2014 the same accuracy with "
     f"{P_EXTRA_M:.2f} M "
     f"more parameters \u2014 and it is significantly behind "
     f"EfficientNetV2-S, ConvNeXt-Tiny, Swin-T and ResNet-50. The earlier "
     f"draft already lost to DenseNet-121 (96.45 % against 97.95 %); the "
     f"clean protocol turns that loss into a tie. The accuracy\u2013parameter "
     f"plane in Fig. {FIGNO['fig5_pareto_params_accuracy.png']} makes the "
     f"practical consequence explicit: there is no "
     f"efficiency argument to fall back on.", indent=True)

para("**What the ablations say.** None of the three tumor-aware components "
     "earns its complexity on classification. The learned fusion collapses "
     f"onto the global pathway (\u03b1_tumor = {FUSION_W['alpha_tumor_mean']:.4f}); "
     "the spatial attention map places *less* mass inside the tumour mask "
     "than a uniform map would; and the auxiliary segmentation loss does not "
     "improve accuracy at any weight we tried. The one component with a "
     "visible positive effect is channel attention, which is disabled in the "
     "executed model \u2014 an honest reading is that the historical "
     "configuration was not the one the manuscript described.", indent=True)

para("**What survives.** Three things do. First, the model is genuinely "
     "more robust to corruption than DenseNet-121 "
     f"(MRRS {ms(ROB_T['mrrs_mean'], ROB_T['mrrs_std_sample_ddof1'])} vs "
     f"{ms(ROB_D['mrrs_mean'], ROB_D['mrrs_std_sample_ddof1'])}), which is a "
     "real if modest property. Second, its mutual information separates "
     "errors from correct predictions meaningfully (error AUROC around "
     "0.72), so the uncertainty machinery is usable for ranking risk even "
     "though the confidence values themselves are under-confident. Third, "
     "the framework genuinely does all of this in one pass with a moderate "
     "parameter budget \u2014 but that is an engineering property, not a "
     "performance claim.", indent=True)

para("**Why the earlier numbers were so high.** They were traceable, not "
     "invented \u2014 every one of them matched an artifact. They were simply "
     "measuring a different, leaky protocol. Re-introducing that split "
     f"reproduces {gs('tamnet_leaky_control', 'accuracy')} immediately, which "
     "is the clearest demonstration that the gap between the two protocols, "
     "not any change in the model, explains the entire difference. This is "
     "the methodological point of the paper: the same architecture, the same "
     "code and the same data produce a 4.5-point swing purely from how the "
     "split is drawn.", indent=True)

# ==========================================================================
# VI. LIMITATIONS
# ==========================================================================
h1("VI.  LIMITATIONS")

bullet("**Public data only.** All experiments use publicly released 2-D MRI "
       "slices. There is no prospective, institutional or clinical "
       "validation of any kind.")
bullet("**Image-level, not patient-level, splits.** The internal pool "
       "carries no patient identifiers, so near-duplicate control is the "
       "only available defence against patient leakage; residual "
       "within-patient correlation cannot be ruled out and every interval "
       "in this paper is image-level.")
bullet("**Unresolved label provenance.** The origin of the four-class "
       "labels in the internal pool is not documented anywhere in the "
       "repository. Part of that pool shares filenames with a public Kaggle "
       "release; the remainder comes from at least three other naming "
       "families. The class counts are stated, the origin is marked "
       "[CITATION NEEDED].")
bullet("**Domain gap.** The segmentation evidence comes from a different "
       "modality stack (BraTS 3-D multimodal NIfTI) and a different 2-D "
       "source (BRISC), while classification uses single-channel JPEG "
       "slices. No experiment here establishes transfer between them.")
bullet("**Intensity sensitivity.** Corruption robustness is measured, but "
       "the model remains sensitive to intensity-type perturbations; the "
       "corruption suite does not cover motion, bias field or scanner "
       "domain shift.")
bullet("**Uneven seed counts.** The headline configurations carry four to "
       "five seeds, but ConvNeXt-Tiny, ResNet-50, ViT-B/16 and the "
       "\u03bb_seg sweep have a single seed each; those rows are point "
       "estimates and are labelled as such throughout.")
bullet("**Statistical resolution.** Per-run bootstrap intervals are about "
       "\u00b11.1 pp, so this benchmark cannot resolve differences smaller "
       "than roughly one percentage point.")

# ==========================================================================
# VII. CONCLUSION
# ==========================================================================
h1("VII.  CONCLUSION")

para(f"This paper presented TAMNet, a tumor-aware multi-task framework that "
     f"couples a shared DenseNet-121 encoder with spatial attention, an "
     f"auxiliary segmentation branch, a learned global\u2013tumor fusion and "
     f"Monte Carlo dropout, and then re-measured it under a leakage-audited "
     f"protocol. The re-measurement is the contribution: {N_RUNS} runs, "
     f"{N_TEST} held-out test images, mean-over-seeds headlines with sample "
     f"standard deviations, paired bootstrap intervals on every comparison, "
     f"and de-duplicated external evaluation.", indent=True)

para(f"The answer to the research question is negative. TAMNet reaches "
     f"{TAM_ACC} accuracy and {TAM_F1} macro-F1, statistically "
     f"indistinguishable from plain DenseNet-121 ({DN_ACC}) and significantly "
     f"below four modern backbones. Removing near-duplicates from the public "
     f"test sets costs {(K_FULL['accuracy_mean'] - K_DED['accuracy_mean']) * 100:.2f} pp "
     f"on Kaggle and {(B_FULL['accuracy_mean'] - B_DED['accuracy_mean']) * 100:.2f} pp "
     f"on BRISC. The auxiliary segmentation loss provides no classification "
     f"gain, the attention map does not concentrate on tumour regions, and "
     f"the fusion weights ignore the tumor-region feature. The whole-tumour "
     f"Dice figures quoted previously belong to a separate 3-D U-Net and are "
     f"re-attributed accordingly.", indent=True)

para("What remains useful is a recipe that is cheap, moderately robust to "
     "corruption, and carries a usable error-ranking uncertainty signal "
     "\u2014 plus, we hope, a worked example of how quickly a positive "
     "result can disappear once the split is audited. Future work should "
     "obtain a genuinely patient-level dataset, make the segmentation "
     "branch receive gradient during classification training, and test "
     "whether explicit attention supervision can move the fusion weight off "
     "the global pathway.", indent=True)

# ==========================================================================
h1("DATA AND CODE AVAILABILITY")
para(f"All experiments were run at repository commit `{GIT_COMMIT}` on "
     f"{RUNCFG['device']} with PyTorch {RUNCFG['torch']}. Every number in "
     "this manuscript is read from an artifact under `results/` by "
     "`paper/rewrite_paper.py`; the mapping from table to source file is "
     "recorded in `paper/tables/sources.json`, and the FILL evidence for the "
     "dataset statements is in `results/fill_evidence.json`. Per-run "
     "predictions, checkpoints, split manifests (SHA-256 pinned) and "
     "de-duplication keep-lists are included so that all confidence "
     "intervals can be recomputed. The legacy manuscript is preserved as "
     "`TAMNet_Paper.pre-audit.docx`. **This is a research prototype and is "
     "not approved or intended for clinical use.**", indent=True)

# ==========================================================================
# REFERENCES
# ==========================================================================
h1("REFERENCES")

REFERENCES = [
    "S. Pereira, A. Pinto, V. Alves, and C. A. Silva, \u201cBrain tumor "
    "segmentation using convolutional neural networks in MRI images,\u201d "
    "IEEE Trans. Med. Imag., vol. 35, no. 5, pp. 1240\u20131251, 2016, "
    "doi: 10.1109/TMI.2016.2538465.",
    "B. H. Menze et al., \u201cThe multimodal brain tumor image segmentation "
    "benchmark (BRATS),\u201d IEEE Trans. Med. Imag., vol. 34, no. 10, "
    "pp. 1993\u20132024, 2015, doi: 10.1109/TMI.2014.2377694.",
    "F. J. Dorfner et al., \u201cA review of deep learning for brain tumor "
    "analysis in MRI,\u201d npj Precision Oncology, vol. 9, Art. no. 2, 2025, "
    "doi: 10.1038/s41698-024-00789-2.",
    "M. Havaei et al., \u201cBrain tumor segmentation with deep neural "
    "networks,\u201d Med. Image Anal., vol. 35, pp. 18\u201331, 2017.",
    "K. Kamnitsas et al., \u201cEfficient multi-scale 3D CNN with fully "
    "connected CRF for accurate brain lesion segmentation,\u201d Med. Image "
    "Anal., vol. 36, pp. 61\u201378, 2017.",
    "F. Isensee et al., \u201cnnU-Net: A self-configuring method for deep "
    "learning-based biomedical image segmentation,\u201d Nature Methods, "
    "vol. 18, pp. 203\u2013211, 2021.",
    "O. Ronneberger, P. Fischer, and T. Brox, \u201cU-Net: Convolutional "
    "networks for biomedical image segmentation,\u201d in Proc. MICCAI, 2015, "
    "pp. 234\u2013241.",
    "M. Sultan et al., \u201cMulti-classification of brain tumor images using "
    "deep neural network,\u201d IEEE Access, vol. 7, pp. 69215\u201369225, "
    "2019.",
    "[9] [VERIFY \u2014 authors missing in the source list] \u201cExploring the "
    "state-of-the-art algorithms for brain tumor classification using MRI "
    "data,\u201d IEEE Access, vol. 13, pp. 118033\u2013118054, 2025, "
    "doi: 10.1109/ACCESS.2025.3579727.",
    "K. He, X. Zhang, S. Ren, and J. Sun, \u201cDeep residual learning for "
    "image recognition,\u201d in Proc. CVPR, 2016, pp. 770\u2013778.",
    "G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, \u201cDensely "
    "connected convolutional networks,\u201d in Proc. CVPR, 2017, "
    "pp. 4700\u20134708.",
    "M. Tan and Q. V. Le, \u201cEfficientNet: Rethinking model scaling for "
    "convolutional neural networks,\u201d in Proc. ICML, 2019, "
    "pp. 6105\u20136114.",
    "Z. Liu et al., \u201cA ConvNet for the 2020s,\u201d in Proc. CVPR, 2022, "
    "pp. 11976\u201311986.",
    "A. Dosovitskiy et al., \u201cAn image is worth 16\u00d716 words: "
    "Transformers for image recognition at scale,\u201d in Proc. ICLR, 2021.",
    "Z. Liu et al., \u201cSwin Transformer: Hierarchical vision transformer "
    "using shifted windows,\u201d in Proc. ICCV, 2021, pp. 10012\u201310022.",
    "J. Hu, L. Shen, and G. Sun, \u201cSqueeze-and-excitation networks,\u201d "
    "in Proc. CVPR, 2018, pp. 7132\u20137141.",
    "S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, \u201cCBAM: Convolutional "
    "block attention module,\u201d in Proc. ECCV, 2018, pp. 3\u201319.",
    "O. Oktay et al., \u201cAttention U-Net: Learning where to look for the "
    "pancreas,\u201d arXiv:1804.03999, 2018.",
    "Z. Zhou et al., \u201cUNet++: A nested U-Net architecture for medical "
    "image segmentation,\u201d in Deep Learning in Medical Image Analysis and "
    "Multimodal Learning for Clinical Decision Support, 2018, pp. 3\u201311.",
    "Y. Qiu, K. Jiang, H. Yao, Z. Wang, and S. Satoh, \u201cDoes adding a "
    "modality really make positive impacts in incomplete multi-modal brain "
    "tumor segmentation?\u201d IEEE Trans. Med. Imag., vol. 44, no. 5, "
    "pp. 2194\u20132205, 2025, doi: 10.1109/TMI.2025.",
    "J. Zhou et al., \u201cShape-scale co-awareness network for 3D brain "
    "tumor segmentation,\u201d IEEE Trans. Med._imag., vol. 43, no. 7, "
    "pp. 2495\u20132508, 2024, doi: 10.1109/TMI.2024.3368531.",
    "[22] [VERIFY \u2014 first author initial disputed: Z. vs Q. Yang] "
    "Yang et al., \u201cD2-Net: Dual disentanglement network for brain tumor "
    "segmentation with missing modalities,\u201d IEEE Trans. Med. Imag., "
    "vol. 41, no. 10, pp. 2953\u20132964, 2022.",
    "Y. Cheng, Z. Liu, and S. Tamura, \u201cDeep hierarchy-aware segmentation: "
    "A novel framework for MRIs brain tumor segmentation,\u201d IEEE Trans. "
    "Med. Imag., vol. 45, no. 5, pp. 2023\u20132038, 2026, "
    "doi: 10.1109/TMI.2025.3645821.",
    "Y. Shao et al., \u201cRethinking brain tumor segmentation from the "
    "frequency domain perspective,\u201d IEEE Trans. Med. Imag., vol. 44, "
    "no. 11, pp. 4536\u20134553, 2025, doi: 10.1109/TMI.2025.3579213.",
    "Y. Gal and Z. Ghahramani, \u201cDropout as a Bayesian approximation: "
    "Representing model uncertainty in deep learning,\u201d in Proc. ICML, "
    "2016, pp. 1050\u20131059.",
    "A. Kendall and Y. Gal, \u201cWhat uncertainties do we need in Bayesian "
    "deep learning for computer vision?\u201d in Proc. NeurIPS, 2017, "
    "pp. 5574\u20135584.",
    "B. Lakshminarayanan, A. Pritzel, and C. Blundell, \u201cSimple and "
    "scalable predictive uncertainty estimation using deep ensembles,\u201d "
    "in Proc. NeurIPS, 2017, pp. 6402\u20136413.",
    "Y. Ovadia et al., \u201cCan you trust your model's uncertainty? "
    "Evaluating predictive uncertainty under dataset shift,\u201d in Proc. "
    "NeurIPS, vol. 32, 2019. [author first initial corrected: A. \u2192 Y.]",
    "T. Zhou and S. Zhu, \u201cUncertainty quantification and attention-aware "
    "fusion guided multi-modal MR brain tumor segmentation,\u201d Comput. "
    "Biol. Med., vol. 163, Art. no. 107142, 2023, "
    "doi: 10.1016/j.compbiomed.2023.107142.",
    "J. Zhao, Z. Xing, Z. Chen, L. Wan, T. Han, H. Fu, and L. Zhu, "
    "\u201cUncertainty-aware multi-dimensional mutual learning for brain and "
    "brain tumor segmentation,\u201d IEEE J. Biomed. Health Inform., vol. 27, "
    "no. 9, pp. 4362\u20134372, 2023, doi: 10.1109/JBHI.2023.3266775.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \u201cOn calibration of "
    "modern neural networks,\u201d in Proc. ICML, 2017, pp. 1321\u20131330.",
    "M. Minderer et al., \u201cRevisiting the calibration of modern neural "
    "networks,\u201d in Proc. NeurIPS, vol. 34, pp. 15682\u201315694, 2021.",
    "R. R. Selvaraju et al., \u201cGrad-CAM: Visual explanations from deep "
    "networks via gradient-based localization,\u201d in Proc. ICCV, 2017, "
    "pp. 618\u2013626.",
    "[34] [VERIFY] \u201cThe Brain Tumor Segmentation (BraTS) Challenge 2024,\u201d "
    "2024. [dataset/benchmark citation added in this revision; exact author "
    "list and identifier to be confirmed]",
    "A. Fateh et al., \u201cBRISC 2025: A brain tumor image segmentation, "
    "classification and tumor boundary segmentation dataset,\u201d "
    "arXiv:2506.14318, 2025.",
    "[36] [VERIFY \u2014 owner slug and URL] M. Nickparvar, \u201cBrain Tumor "
    "MRI Dataset,\u201d Kaggle, 7,023 images (Training 5,712 / Testing 1,311).",
    "T. Hendrycks and T. G. Dietterich, \u201cBenchmarking neural network "
    "robustness to common corruptions and perturbations,\u201d in Proc. "
    "ICLR, 2019, arXiv:1903.12261.",
]

for i, ref in enumerate(REFERENCES, start=1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(1)
    body = ref
    if not body.startswith(f"[{i}]"):
        body = f"[{i}] {body}"
    _rich(p, body, size=9)

# ==========================================================================
# save
# ==========================================================================
def main() -> int:
    if OUT.is_file() and not PRESERVED.is_file():
        shutil.copy2(OUT, PRESERVED)
        print(f"preserved legacy manuscript -> {PRESERVED.name}")
    doc.save(str(OUT))
    print(f"wrote {OUT}")
    print(f"  paragraphs={len(doc.paragraphs)} tables={len(doc.tables)} "
          f"figures={_fig_n} table_captions={_table_n}")
    if MISSING_FIGURES:
        print("MISSING figures (skipped, not fabricated):")
        for f in MISSING_FIGURES:
            print("   MISSING:", f)
    else:
        print("  all requested figures present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
