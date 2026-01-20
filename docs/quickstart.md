# Quickstart

## 1) Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run the full pipeline (using default inputs)

From the repo root:

```bash
python src/explain_omics_pipeline.py
```

Outputs are written to:

```
results/explain_omics/
  csv/
  figures_png/
  evaluation/
```

## 3) Run using the included small demo inputs

```bash
python src/explain_omics_pipeline.py --inputs data/example_inputs --outdir results/demo_run
```

## 4) Query with partial inputs (inference layer)

After you have pipeline outputs (CSV tables), you can query the ranked chains.

```bash
python src/explain_omics_inference.py \
  --csv_dir results/explain_omics/csv \
  --microbes Fusobacterium,Lactobacillus \
  --top_k 20
```

Or pathway-only:

```bash
python src/explain_omics_inference.py \
  --csv_dir results/explain_omics/csv \
  --pathways "Glycerophospholipid metabolism" \
  --top_k 20
```

