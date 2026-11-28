# ---------------------------------------------------------------------------
# TAMNet audit — reproduction pipeline.  `make help` lists the stages.
# Same order as run_all.sh / run_all.ps1; every stage is skip-if-done, so
# `make all` on a finished tree re-runs only the cheap paper/QA stages.
#
#   make install     # pip install -r requirements.txt
#   make all         # split -> ... -> paper -> check
#   make check       # QA only (numbers, figures, unit tests)
#
# Windows note: this repo was audited on PowerShell; `run_all.ps1` is the
# native equivalent of these targets.
# ---------------------------------------------------------------------------
PY      ?= python
PYTHONX = $(PY) -X utf8

.PHONY: all install split train evaluate aggregate bootstrap external \
        uncertainty robustness seg brats3d attention efficiency qual \
        figures tables paper check test clean-figures help

all: split train evaluate aggregate bootstrap external uncertainty robustness \
     seg brats3d attention efficiency qual figures tables paper check

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:            ## pip install -r requirements.txt
	$(PY) -m pip install -r requirements.txt

split:              ## manifest-pinned split + leakage audit
	$(PYTHONX) scripts/phase3_split.py
train:              ## 41 runs x 30 epochs (skips finished runs)
	$(PYTHONX) scripts/phase3_train.py
evaluate:           ## TEST eval, one per configuration
	$(PYTHONX) scripts/phase3_evaluate.py
aggregate:          ## grouped mean/std over seeds (ddof=1)
	$(PYTHONX) scripts/phase3_aggregate.py
bootstrap:          ## bootstrap 95 % CIs + paired deltas (B=1000)
	$(PYTHONX) scripts/phase3_bootstrap.py
external:           ## external eval, full + de-duplicated
	$(PYTHONX) scripts/phase3_external_eval.py
uncertainty:        ## reliability, ECE, AURC, risk-coverage
	$(PYTHONX) scripts/phase3_uncertainty.py
robustness:         ## 27 corruptions x 3 severities
	$(PYTHONX) scripts/phase3_robustness.py
seg:                ## BRISC auxiliary-head Dice
	$(PYTHONX) scripts/phase3_seg_eval.py
brats3d:            ## standalone 3-D U-Net -> WT/TC/ET (D3) + ingest
	$(PYTHONX) -m scripts.evaluate_segmentation --split test
	$(PYTHONX) scripts/phase5_ingest_unet3d.py
attention:          ## attention alignment + Grad-CAM (missing is recorded)
	$(PYTHONX) scripts/phase3_attention_alignment.py
efficiency:         ## params / MiB / single-pass and T=10 MC latency
	$(PYTHONX) scripts/phase3_efficiency.py
qual:               ## fig7 qualitative + per-sample preds + supplementary S1-S10
	$(PYTHONX) scripts/phase4_seg_qualitative.py
	$(PYTHONX) scripts/phase4_save_predictions.py
	$(PYTHONX) scripts/make_supplementary.py
figures:            ## fig1-16, vector PDF + 600-dpi PNG
	$(PYTHONX) scripts/make_figures.py
	$(PYTHONX) scripts/make_figures_static.py
tables:             ## 12 tables from results/ only
	$(PYTHONX) scripts/make_tables.py
paper:              ## regenerate TAMNet_Paper.docx from results/
	$(PYTHONX) paper/rewrite_paper.py
check:              ## every number traces to results/ + structure + tests
	$(PYTHONX) scripts/check_paper_numbers.py
	$(PYTHONX) scripts/check_figures.py
	$(PYTHONX) -m pytest tests -q
test:               ## unit tests only
	$(PYTHONX) -m pytest tests -q
clean-figures:      ## move figure outputs aside (never deletes)
	@mkdir -p paper/figs/aside
	-@mv paper/figs/*.png paper/figs/*.pdf paper/figs/aside/ 2>/dev/null || true
