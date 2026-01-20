# Output files

All outputs are written under:

```
results/explain_omics/
```

## csv/

Key tables you will typically use:

- **HostModule_Reference.csv** — module labels and weights.
- **Obj2_Pathways_CLEAN.csv** — cleaned pathway evidence table.
- **Microbes_withScore.csv** — microbes + `MicrobeScore`.
- **Microbe_to_Pathway_Links.csv** — microbe→pathway edges (`LinkScore`).
- **Pathway_to_HostModule_Links.csv** — aligned pathway→module edges (`Match_Score`, `Evidence_Score_Obj2`).
- **Triplets_HostModule_Pathway_Microbe.csv** — ranked host–pathway–microbe chains (`ChainScore`, `TripletConfidence`).
- **Chains4_Microbe_Pathway_HostModule_Gene.csv** — optional 4-layer chains if `genes` are present.
- **NodeRankings_GraphAI_Consensus.csv** — PageRank/centralities + consensus driver ranking.
- **NodeEmbeddings_RandomWalkSVD.csv** — learned embeddings per node.
- **PredictedLinks_*.csv** — graph-ML hypothesis links not in the original network.

## figures_png/

- `08_MODEL_Pipeline_Schematic.png` — pipeline schematic.
- `11_LayeredNetwork.png` — layered network overview.
- `15_Network_3Layer_*.png` — readability-optimized 3-layer interaction network.
- `16_Network_4Layer_*.png` — readability-optimized 4-layer interaction network.

## evaluation/

- `Evaluation_LinkPrediction_AUC_AP.csv` — evaluation of link prediction using edge holdout.
- `BestModel_Report.csv` — summary report.

## OutputManifest.csv
A machine-readable list of all generated files.
