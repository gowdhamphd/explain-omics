# Input file formats (CSV)

`explain-omics` runs on **processed result tables** from your upstream analyses (e.g., microbiome feature selection, metabolomic pathway results, host-module results). It does **not** require raw sample-level matrices.

The pipeline expects **5 CSV files**.

## A) Microbiome features
**Default:** `data/inputs/microbiome_features.csv`

Required columns
- `Genus` (string)
- At least one of:
  - Evidence/importance style columns (any column name containing `Importance`, `Hub`, or `Evidence`), **or**
  - Rank columns (any column name containing `Rank`)
- Differential direction columns used to connect microbes to the pathway comparisons:
  - `DE_Tumor_vs_Adjacent` for `Comparison = Tumor_vs_Adjacent`
  - `DE_Metastatic_vs_NonMet` for `Comparison = Metastatic_vs_NonMet`

Notes
- Direction columns can be strings such as `Up`, `Down`, `Higher`, `Lower` (the pipeline stores them for reporting; it does not enforce a strict vocabulary).

Example (minimal)
```csv
Genus,DE_Tumor_vs_Adjacent,RF_TissueType_Rank,RF_TissueType_Importance
Fusobacterium,Up,3,0.031
Lactobacillus,Down,8,0.010
```

## B) Metabolomic pathway mediators
**Default:** `data/inputs/metabolomic_pathways.csv`

Required columns
- `Feature_Name` (string) — pathway name
- `Comparison` (string) — e.g., `Tumor_vs_Adjacent`, `Metastatic_vs_NonMet`
- `Functional_Class` (string)
- `Evidence_Score` (numeric)

Optional columns (kept for reporting if present)
- `FDR` (numeric)
- `Hits` (numeric)

Example (minimal)
```csv
Feature_Name,Comparison,Functional_Class,Evidence_Score
Glycerophospholipid metabolism,Tumor_vs_Adjacent,Lipid metabolism,4.2
Steroid hormone biosynthesis,Metastatic_vs_NonMet,Hormone metabolism,3.6
```

## C) Host module summary
**Default:** `data/inputs/host_modules_summary.csv`

Required columns
- `host_module` (string)

Recommended columns
- `n_pathways` (numeric) — used to weight modules (`ModuleWeight = log1p(n_pathways)`)

Example (minimal)
```csv
host_module,n_pathways
Layer1__Immune_Signaling,45
Layer2__ECM_Remodeling,18
```

## D) Host module → pathway table
**Default:** `data/inputs/host_modules_all_pathways.csv`

Required columns
- `host_module` (string)
- `pathway` (string)

Optional but recommended (enables gene-layer & improves interpretability)
- `genes` — semicolon-separated gene list, e.g. `TP53;BRCA1;...`
- `p.adjust` — pathway adjusted p-value (used to weight host→gene edges)

Example (minimal)
```csv
host_module,pathway,genes,p.adjust
Layer1__Immune_Signaling,Cytokine signaling in immune system,IL6;STAT3;JAK1,1e-6
```

## E) Host module counts
**Default:** `data/inputs/host_modules_counts.csv`

Required columns
- `host_module` (string)

Optional
- `n_sig_pathways` (numeric) — used for reporting/annotation

Example (minimal)
```csv
host_module,n_sig_pathways
Layer1__Immune_Signaling,12
Layer2__ECM_Remodeling,5
```

---

## Quick validation checklist
Before running, confirm:
- `Feature_Name` values in metabolomics are reasonably similar to `pathway` values in host-module pathway table (fuzzy matching is used).
- Your metabolomics `Comparison` values align with the microbiome direction columns you have.

