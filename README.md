# explain-omics

**explain-omics** is an **explainable, network-based multi-omics interaction prioritization tool** that integrates:
- tissue microbiome genera (processed feature/evidence table)
- metabolomic pathways (processed pathway evidence table)
- host functional modules (processed module + pathway/gene tables)

It produces ranked **microbe → pathway → host-module** interaction chains (and optional gene-layer chains), plus graph-AI driver rankings and graph-ML hypothesis link prediction.

> **Note:** This is **not** a clinical prediction model. Evaluation is performed as **held-out edge reconstruction** (AUC/AP) in the constructed network.

## Quickstart

- Read: `docs/quickstart.md`
- Input schemas: `docs/input_formats.md`
- Output guide: `docs/outputs.md`
- Tutorial: `docs/tutorial.md`

### Install

```bash
pip install -r requirements.txt
```

### Run with your inputs

```bash
python src/explain_omics_pipeline.py
```

Outputs:

```
results/explain_omics/
```

### Run demo inputs

```bash
python src/explain_omics_pipeline.py \
  --inputs data/example_inputs \
  --outdir results/example_run
```

## Partial-input inference (after you run the pipeline)

```bash
python src/explain_omics_inference.py \
  --csv_dir results/explain_omics/csv \
  --microbes "Fusobacterium" "Lactobacillus" \
  --top_k 20
```

Notebook:
- `notebooks/01_inference_demo.ipynb`

## License
MIT (see `LICENSE`).
