# Example inputs

This folder contains **small demo CSVs** (subsets of the full inputs) so anyone can run `explain-omics` quickly.

## Run the pipeline on demo inputs

From the repo root:

```bash
python src/explain_omics_pipeline.py \
  --inputs data/example_inputs \
  --outdir results/example_run
```

## Demo files

- `microbiome_features_demo.csv`  
  25 genera with evidence/rank columns and DE direction columns.

- `metabolomic_pathways_demo.csv`  
  12 pathway rows with the required columns (`Feature_Name`, `Comparison`, `Functional_Class`, `Evidence_Score`).

- `host_modules_summary_demo.csv`  
  Module summary rows for the modules referenced in the demo pathway table.

- `host_modules_all_pathways_demo.csv`  
  A minimal host-module pathway table designed to *match* the demo metabolomic pathways (so fuzzy matching succeeds).

- `host_modules_counts_demo.csv`  
  Counts table for the demo modules.

## Notes
- These demo files are intended only for testing the workflow and generating example outputs.
- For real analyses, replace the files under `data/inputs/` with your full processed results.
