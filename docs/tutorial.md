# Tutorial: from inputs to ranked interaction chains

This tutorial explains what the pipeline computes and how to use partial inputs for hypothesis generation.

## Step 0 — Prepare the 5 input tables
See `docs/input_formats.md` for schemas.

Recommended workflow:
1. **Microbiome features**: genus-level table that includes evidence / ranks from your upstream Obj-1 analysis.
2. **Metabolomic pathways**: pathway-level table with an `Evidence_Score` (e.g., pathway enrichment score, multivariate contribution, ...)
3. **Host modules**: module summary + module→pathway table from your upstream Obj-3 analysis.

Place them under `data/inputs/` using the standard filenames, or pass `--inputs_dir` to the pipeline.

## Step 1 — Run the pipeline

```bash
python src/explain_omics_pipeline.py
```

What happens inside:
- **Pathway alignment (NLP fuzzy matching):** metabolomics pathways are aligned to host-module pathway names.
- **Explainable scoring:**
  - `MicrobeScore` from upstream microbe evidence
  - `LinkScore = MicrobeScore × Evidence_Score(pathway)`
  - `ChainScore = LinkScore × Match_Score × ModuleWeight`
- **Graph construction:** Microbe–Pathway–HostModule (+ optional Gene) heterogeneous graph
- **Graph-AI ranking:** PageRank + centralities + consensus ranking
- **Graph-ML embeddings:** random-walk co-occurrence + SVD
- **Link prediction + evaluation:** edge holdout, AUC + AP (network reconstruction; not clinical prediction)

## Step 2 — Inspect the key outputs

### Most important CSVs
- `results/explain_omics/csv/Triplets_HostModule_Pathway_Microbe.csv`
- `results/explain_omics/csv/Chains4_Microbe_Pathway_HostModule_Gene.csv` (only if `genes` exist)
- `results/explain_omics/csv/NodeRankings_GraphAI_Consensus.csv`
- `results/explain_omics/evaluation/Evaluation_LinkPrediction_AUC_AP.csv`

### Figures to look at
- `figures_png/06_TopChains.png` (top 3-layer chains)
- `figures_png/07_TopChains4_Gene.png` (top 4-layer chains, if available)
- `figures_png/11_LayeredNetwork.png` (layered network view)
- `evaluation/Evaluation_AUC.png` and `evaluation/Evaluation_AP.png`

## Step 3 — Use partial inputs (inference)

The inference tool does **not retrain** anything. It filters and ranks from the generated CSV tables.

### Microbe-only input
Example: *“Given these microbes, what are the most likely pathways, host modules, and genes?”*

```bash
python src/explain_omics_inference.py \
  --csv_dir results/explain_omics/csv \
  --microbes "Fusobacterium" "Lactobacillus" \
  --top_k 25
```

### Pathway-only input
Example: *“Given this pathway, what host programs/modules are prioritized?”*

```bash
python src/explain_omics_inference.py \
  --csv_dir results/explain_omics/csv \
  --pathways "Glycerophospholipid metabolism" \
  --top_k 25
```

### Microbes + pathways (most specific)

```bash
python src/explain_omics_inference.py \
  --csv_dir results/explain_omics/csv \
  --microbes "Fusobacterium" \
  --pathways "Glycerophospholipid metabolism" \
  --top_k 25
```

## Troubleshooting

### “No Obj2→Obj3 matches found”
- Ensure pathway names between metabolomics and host-module tables are reasonably similar.
- Install `rapidfuzz` (improves fuzzy matching quality): `pip install rapidfuzz`.

### “No Microbe→Pathway links created”
- Check that your microbiome input contains the right direction columns for the comparisons present in metabolomics.

