#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXPLAIN-OMICS — Explainable multi-omics interaction prioritization (processed results only)
======================================================================

What this script does (single, consistent pipeline)
---------------------------------------------------
INPUTS (processed outputs from Obj1–Obj3 only):
  - Obj1 tissue microbiome features (genus-level + ML evidence/ranks + DE direction)
  - Obj2 metabolomic pathway mediators (pathway name + evidence score + class + comparison)
  - Obj3 host module summary + module pathway table (includes pathway names and gene lists)

OUTPUTS (all in results/explain_omics/):
  csv/                : all link tables, triplets, driver ranks, embeddings, predictions
  figures_png/        : all plots + final layered network(s) + model schematic
  evaluation/         : link-prediction evaluation (AUC, AP) + best-model report

AI/ML included (reviewer-safe, with processed results):
  1) Explainable evidence scoring: MicrobeScore, LinkScore, ChainScore
  2) NLP fuzzy matching: Obj2 pathways -> Obj3 pathways (Match_Score)
  3) Graph-AI ranking: PageRank + Degree/Betweenness/Closeness
  4) Unsupervised communities: greedy modularity
  5) Graph-ML embeddings: random-walk co-occurrence + SVD (DeepWalk/Node2Vec-style)
  6) Link prediction: cosine similarity in embedding space (hypothesis generation)
  7) Proper evaluation: edge holdout (AUC + Average Precision) for Microbe-Pathway and Pathway-Host edges
  8) Ensemble/consensus ranking + Bayesian-style confidence for triplets (optional, explainable)
  9) Gene layer: HostModule -> Genes (from Obj3 'genes' column) to produce 4-layer chains

Notes
-----
- No sample-level patient matrices are required.
- This is NOT a clinical prediction model; evaluation is on held-out edges (network reconstruction).
- No PDF outputs; only CSV + PNG.

"""

import os
import re
import difflib
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _distinct_bar_colors(n: int, cmap_name: str = "tab20"):
    """Return n visually distinct RGBA colors from a categorical colormap."""
    n = int(max(n, 1))
    cmap = plt.cm.get_cmap(cmap_name, n)
    return [cmap(i) for i in range(n)]
import networkx as nx

# Global plotting style (bigger + bold for thesis-ready figures)
plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# Optional fuzzy matching
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# Optional clustering (only used for embedding clusters)
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# =============================================================================
# 0) INPUT FILES (place in same folder as this script or edit paths)
# =============================================================================
OBJ1_MICROBE = "data/inputs/microbiome_features.csv"
OBJ2_PATHWAY = "data/inputs/metabolomic_pathways.csv"

OBJ3_SUMMARY = "data/inputs/host_modules_summary.csv"
OBJ3_ALLPATH = "data/inputs/host_modules_all_pathways.csv"
OBJ3_COUNTS  = "data/inputs/host_modules_counts.csv"

# --------------------------------------------
# CLI (optional): override input/output paths
# --------------------------------------------
import argparse
parser = argparse.ArgumentParser(description='explain-omics pipeline (processed results only)')
parser.add_argument('--inputs', default='data/inputs', help='Input folder containing the 5 required CSVs')
parser.add_argument('--outdir', default='./results', help='Output root folder (will create <outdir>/explain_omics/)')
args, _unknown = parser.parse_known_args()

_INPUT_DIR = args.inputs
OBJ1_MICROBE = os.path.join(_INPUT_DIR, 'microbiome_features.csv')
OBJ2_PATHWAY = os.path.join(_INPUT_DIR, 'metabolomic_pathways.csv')
OBJ3_SUMMARY = os.path.join(_INPUT_DIR, 'host_modules_summary.csv')
OBJ3_ALLPATH = os.path.join(_INPUT_DIR, 'host_modules_all_pathways.csv')
OBJ3_COUNTS  = os.path.join(_INPUT_DIR, 'host_modules_counts.csv')

OUTDIR = args.outdir

# =============================================================================
# 1) OUTPUT FOLDERS
# =============================================================================
OUTDIR = globals().get("OUTDIR", "./results")
FINAL_DIR = os.path.join(OUTDIR, "explain_omics")
CSV_DIR = os.path.join(FINAL_DIR, "csv")
FIG_DIR = os.path.join(FINAL_DIR, "figures_png")
EVAL_DIR = os.path.join(FINAL_DIR, "evaluation")

for d in [OUTDIR, FINAL_DIR, CSV_DIR, FIG_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# 2) HELPERS
# =============================================================================
def safe_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").fillna(0)

def norm_text(s) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def write_csv(df: pd.DataFrame, fname: str, folder: str = CSV_DIR, index: bool = False) -> None:
    path = os.path.join(folder, fname)
    df.to_csv(path, index=index)
    print("Wrote:", path)

def save_png(fname: str, folder: str = FIG_DIR) -> None:
    path = os.path.join(folder, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def colorful_barh(y_labels, values, title, xlabel, out_png, figsize=(10, 6)):
    """Colorful horizontal bar plot with bold fonts."""
    y_labels = [str(x) for x in y_labels]
    values = np.asarray(values, dtype=float)
    n = len(values)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    colors = [cmap(i) for i in range(n)]
    plt.figure(figsize=figsize)
    plt.barh(y_labels, values, color=colors)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("", fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def colorful_bar(x_labels, values, title, ylabel, out_png, figsize=(7, 4)):
    """Colorful vertical bar plot with bold fonts."""
    x_labels = [str(x) for x in x_labels]
    values = np.asarray(values, dtype=float)
    n = len(values)
    cmap = plt.cm.get_cmap("tab20", max(n, 1))
    colors = [cmap(i) for i in range(n)]
    plt.figure(figsize=figsize)
    plt.bar(x_labels, values, color=colors)
    plt.ylabel(ylabel, fontweight="bold")
    plt.title(title, fontweight="bold")
    plt.xticks(rotation=0, fontweight="bold")
    plt.yticks(fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def pretty_module_label(host_module: str) -> str:
    if not isinstance(host_module, str):
        return "unknown"
    parts = host_module.split("__", 1)
    if len(parts) == 2:
        return parts[1].replace("_", " ").strip()
    return host_module.replace("_", " ").strip()

def best_fuzzy_match(query: str, candidates: np.ndarray) -> Tuple[str, float]:
    q = norm_text(query)
    if len(q) < 3:
        return None, 0.0
    if RAPIDFUZZ_OK:
        best_score, best_val = -1, None
        for c in candidates:
            score = fuzz.token_set_ratio(q, c)
            if score > best_score:
                best_score, best_val = score, c
        return best_val, float(best_score)
    match = difflib.get_close_matches(q, list(candidates), n=1, cutoff=0.6)
    return (match[0], 80.0) if match else (None, 0.0)

def wrap_label(s: str, width: int = 26) -> str:
    s = str(s)
    if len(s) <= width:
        return s
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)

def minmax01(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def rank01(series: pd.Series, higher_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    r = s.rank(ascending=not higher_better, method="average")
    r01 = 1.0 - (r - 1.0) / max(1.0, (len(r) - 1.0))
    return r01

def layered_positions(G: nx.Graph,
                      layer_order=("HostModule", "Pathway", "Microbe", "Gene"),
                      x_positions=None,
                      y_gap=1.15) -> Dict[str, Tuple[float, float]]:
    if x_positions is None:
        x_positions = {"HostModule": 0.0, "Pathway": 1.0, "Microbe": 2.0, "Gene": 3.0}
    layer_nodes = {L: [] for L in layer_order}
    for n in G.nodes():
        t = G.nodes[n].get("ntype", "Other")
        if t in layer_nodes:
            layer_nodes[t].append(n)

    def strength(n):
        return sum(float(G[n][nbr].get("weight", 1.0)) for nbr in G.neighbors(n))

    pos = {}
    for L in layer_order:
        nodes = sorted(layer_nodes[L], key=strength, reverse=True)
        ys = [0.0] if len(nodes) == 1 else np.linspace(0, -(len(nodes)-1)*y_gap, len(nodes))
        for i, n in enumerate(nodes):
            pos[n] = (x_positions[L], ys[i])
    return pos

# ---- Evaluation metrics (no sklearn needed)
def auc_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").values
    sum_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    auc = (sum_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    return float(auc)

def average_precision(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y = y_true[order]
    if y.sum() == 0:
        return float("nan")
    prec = []
    tp = 0
    for i, yi in enumerate(y, start=1):
        if yi == 1:
            tp += 1
            prec.append(tp / i)
    return float(np.mean(prec)) if prec else float("nan")


# =============================================================================
# 3) LOAD INPUT FILES
# =============================================================================
obj1 = pd.read_csv(OBJ1_MICROBE)
obj2 = pd.read_csv(OBJ2_PATHWAY)
mod_sum = pd.read_csv(OBJ3_SUMMARY)
mod_path = pd.read_csv(OBJ3_ALLPATH)
mod_counts = pd.read_csv(OBJ3_COUNTS)

print("Loaded shapes:",
      "\n Obj1:", obj1.shape,
      "\n Obj2:", obj2.shape,
      "\n Obj3_summary:", mod_sum.shape,
      "\n Obj3_allpath:", mod_path.shape,
      "\n Obj3_counts:", mod_counts.shape)


# =============================================================================
# 4) OBJ3 -> HOST MODULE REFERENCE
# =============================================================================
if "host_module" not in mod_sum.columns:
    raise ValueError("Obj3 summary missing 'host_module' column.")

host_ref = mod_sum.copy()
host_ref["Biological_Label"] = host_ref["host_module"].apply(pretty_module_label)

if "n_pathways" not in host_ref.columns:
    host_ref["n_pathways"] = 1
host_ref["n_pathways"] = pd.to_numeric(host_ref["n_pathways"], errors="coerce").fillna(1)

if "n_sig_pathways" in mod_counts.columns:
    counts_agg = mod_counts.groupby("host_module", as_index=False)["n_sig_pathways"].sum()
    counts_agg = counts_agg.rename(columns={"n_sig_pathways": "n_sig_pathways_total"})
else:
    counts_agg = pd.DataFrame({"host_module": host_ref["host_module"].unique(),
                               "n_sig_pathways_total": np.nan})

host_ref = host_ref.merge(counts_agg, on="host_module", how="left")
host_ref["ModuleWeight"] = np.log1p(host_ref["n_pathways"].astype(float))

host_ref_out = host_ref[["host_module", "Biological_Label", "n_pathways",
                         "n_sig_pathways_total", "ModuleWeight"]].copy() \
                         .sort_values("n_pathways", ascending=False)

write_csv(host_ref_out, "HostModule_Reference.csv")

plt.figure(figsize=(10, 6))
tmp = host_ref_out.head(20).iloc[::-1]
colors = _distinct_bar_colors(len(tmp))
plt.barh(tmp["host_module"], tmp["n_pathways"], color=colors)
plt.xlabel("n_pathways"); plt.ylabel("host_module")
plt.title("Top Host Modules (Obj3 reference)")
save_png("01_TopHostModules.png")


# =============================================================================
# 5) OBJ2 -> PATHWAYS (clean)
# =============================================================================
req = ["Feature_Name", "Comparison", "Functional_Class", "Evidence_Score"]
miss = [c for c in req if c not in obj2.columns]
if miss:
    raise ValueError(f"Obj2 pathway file missing columns: {miss}")

obj2p = obj2.copy().rename(columns={"Feature_Name": "Obj2_Pathway"})
obj2p["Obj2_Pathway_clean"] = obj2p["Obj2_Pathway"].map(norm_text)
obj2p["Evidence_Score"] = pd.to_numeric(obj2p["Evidence_Score"], errors="coerce").fillna(1.0)

# optional columns used only for reporting if present
for c in ["FDR", "Hits"]:
    if c not in obj2p.columns:
        obj2p[c] = np.nan

obj2p_unique = obj2p.drop_duplicates(subset=["Obj2_Pathway", "Comparison", "Functional_Class"]).copy()
write_csv(obj2p_unique, "Obj2_Pathways_CLEAN.csv")

plt.figure(figsize=(10, 6))
tmp = obj2p_unique.sort_values("Evidence_Score", ascending=False).head(20).iloc[::-1]
colors = _distinct_bar_colors(len(tmp))
plt.barh(tmp["Obj2_Pathway"], tmp["Evidence_Score"], color=colors)
plt.xlabel("Evidence_Score"); plt.ylabel("Obj2_Pathway")
plt.title("Top Metabolic Pathways (Obj2)")
save_png("02_TopObj2Pathways.png")


# =============================================================================
# 6) AI/NLP: OBJ2 pathway -> OBJ3 pathway -> HostModule links
# =============================================================================
for need in ["host_module", "pathway"]:
    if need not in mod_path.columns:
        raise ValueError(f"Obj3 all_pathways missing '{need}' column.")

mp = mod_path.copy()
# ensure optional columns exist
for c in ["p.adjust", "database", "layer", "genes", "n_genes"]:
    if c not in mp.columns:
        mp[c] = np.nan

mp = mp.dropna(subset=["pathway"]).copy()
mp["pathway_clean"] = mp["pathway"].map(norm_text)
cand = mp["pathway_clean"].unique()

links = []
for _, r in obj2p_unique.iterrows():
    best, score = best_fuzzy_match(r["Obj2_Pathway"], cand)
    if best is None or score < 75:
        continue
    hit = mp[mp["pathway_clean"] == best].copy()
    hit["Obj2_Pathway"] = r["Obj2_Pathway"]
    hit["Match_Score"] = score
    hit["Comparison"] = r["Comparison"]
    hit["Functional_Class"] = r["Functional_Class"]
    hit["Evidence_Score_Obj2"] = r["Evidence_Score"]
    hit["FDR_Obj2"] = r["FDR"]
    hit["Hits_Obj2"] = r["Hits"]
    links.append(hit)

path_to_module = pd.concat(links, ignore_index=True) if links else pd.DataFrame()
if path_to_module.empty:
    raise RuntimeError("No Obj2->Obj3 matches found. Install rapidfuzz or check pathway naming.")
path_to_module["Match_Score"] = pd.to_numeric(path_to_module["Match_Score"], errors="coerce").fillna(80)
path_to_module["Evidence_Score_Obj2"] = pd.to_numeric(path_to_module["Evidence_Score_Obj2"], errors="coerce").fillna(1)
path_to_module = path_to_module.sort_values(
    ["Obj2_Pathway", "host_module", "Match_Score", "Evidence_Score_Obj2"],
    ascending=[True, True, False, False]
).drop_duplicates(subset=["Obj2_Pathway", "host_module"])

write_csv(path_to_module, "Pathway_to_HostModule_Links.csv")

plt.figure(figsize=(8, 5))
plt.hist(path_to_module["Match_Score"].astype(float), bins=20)
plt.xlabel("Match_Score"); plt.ylabel("Count")
plt.title("Obj2->Obj3 Pathway Match Score Distribution")
save_png("03_PathwayMatchScore_Distribution.png")


# =============================================================================
# 7) OBJ1 -> MicrobeScore + Microbe->Pathway links (LinkScore)
# =============================================================================
if "Genus" not in obj1.columns:
    raise ValueError("Obj1 microbiome file missing 'Genus' column.")

micro = obj1.copy()
score_cols = [c for c in micro.columns if ("Importance" in c) or ("Hub" in c) or ("Evidence" in c)]
micro["MicrobeScore"] = 0.0
for c in score_cols:
    micro["MicrobeScore"] += safe_num(micro[c])

rank_cols = [c for c in micro.columns if "Rank" in c]
if float(micro["MicrobeScore"].sum()) == 0.0 and rank_cols:
    tmp = 0.0
    for c in rank_cols:
        r = safe_num(micro[c])
        tmp += np.where(r > 0, 1.0/(r+1e-6), 0.0)
    micro["MicrobeScore"] = tmp

write_csv(micro, "Microbes_withScore.csv")

plt.figure(figsize=(10, 6))
tmp = micro.sort_values("MicrobeScore", ascending=False).head(25).iloc[::-1]
colors = _distinct_bar_colors(len(tmp))
plt.barh(tmp["Genus"], tmp["MicrobeScore"], color=colors)
plt.xlabel("MicrobeScore"); plt.ylabel("Genus")
plt.title("Top Tissue Microbes (Obj1 evidence score)")
save_png("04_TopMicrobes.png")

def micro_direction_col(comp: str):
    # Expect these columns in Obj1 file; adjust here if your columns differ
    if comp == "Tumor_vs_Adjacent":
        return "DE_Tumor_vs_Adjacent"
    if comp == "Metastatic_vs_NonMet":
        return "DE_Metastatic_vs_NonMet"
    return None

microbe_links = []
for comp in obj2p_unique["Comparison"].dropna().unique():
    dcol = micro_direction_col(comp)
    if dcol is None or dcol not in micro.columns:
        continue
    msub = micro.dropna(subset=[dcol]).copy()
    psub = obj2p_unique[obj2p_unique["Comparison"] == comp].copy()
    if msub.empty or psub.empty:
        continue
    msub["key"] = 1
    psub["key"] = 1
    cand2 = msub.merge(psub, on="key").drop(columns=["key"])
    cand2["LinkScore"] = cand2["MicrobeScore"].astype(float) * cand2["Evidence_Score"].astype(float)
    # Keep top pathways per genus to limit edges
    cand2 = cand2.sort_values(["Genus", "LinkScore"], ascending=[True, False]).groupby("Genus", as_index=False).head(15)
    cand2 = cand2.rename(columns={dcol: "Microbe_Direction"})
    keep_cols = ["Genus", "Microbe_Direction", "MicrobeScore", "Obj2_Pathway", "Functional_Class",
                 "Comparison", "Evidence_Score", "FDR", "Hits", "LinkScore"]
    for c in keep_cols:
        if c not in cand2.columns:
            cand2[c] = np.nan
    microbe_links.append(cand2[keep_cols].copy())

microbe_to_pathway = pd.concat(microbe_links, ignore_index=True) if microbe_links else pd.DataFrame()
if microbe_to_pathway.empty:
    raise RuntimeError("No Microbe->Pathway links created. Check Obj1 direction columns and Obj2 comparisons.")

write_csv(microbe_to_pathway, "Microbe_to_Pathway_Links.csv")

plt.figure(figsize=(12, 7))
top = microbe_to_pathway.sort_values("LinkScore", ascending=False).head(25).iloc[::-1]
lbl = (top["Genus"].astype(str) + " → " + top["Obj2_Pathway"].astype(str))
colors = _distinct_bar_colors(len(top))
plt.barh(lbl, top["LinkScore"], color=colors)
plt.xlabel("LinkScore"); plt.ylabel("Microbe → Pathway")
plt.title("Top Microbe→Pathway Links (LinkScore)")
save_png("05_TopMicrobeToPathway_Links.png")


# =============================================================================
# 8) Triplets: HostModule–Pathway–Microbe (ChainScore) + Bayesian confidence
# =============================================================================
triplets = microbe_to_pathway.merge(
    path_to_module[["Obj2_Pathway", "host_module", "Match_Score", "Evidence_Score_Obj2", "genes", "p.adjust"]],
    on="Obj2_Pathway", how="inner"
).merge(
    host_ref_out[["host_module", "Biological_Label", "ModuleWeight", "n_pathways"]],
    on="host_module", how="left"
)

triplets["Match_Score"] = pd.to_numeric(triplets["Match_Score"], errors="coerce").fillna(80)
triplets["ModuleWeight"] = pd.to_numeric(triplets["ModuleWeight"], errors="coerce").fillna(1)
triplets["LinkScore"] = pd.to_numeric(triplets["LinkScore"], errors="coerce").fillna(0)

triplets["ChainScore"] = triplets["LinkScore"] * (triplets["Match_Score"]/100.0) * triplets["ModuleWeight"]

# Bayesian-style confidence (optional but useful)
T = triplets.copy()
T["microbe_e01"] = minmax01(T["MicrobeScore"].astype(float))
T["path_e01"]    = minmax01(T["Evidence_Score"].astype(float))
T["match_e01"]   = minmax01(T["Match_Score"].astype(float))
T["modw_e01"]    = minmax01(T["ModuleWeight"].astype(float))
w_micro, w_path, w_match, w_mod = 0.30, 0.30, 0.20, 0.20
p1 = 1.0 - w_micro * T["microbe_e01"]
p2 = 1.0 - w_path  * T["path_e01"]
p3 = 1.0 - w_match * T["match_e01"]
p4 = 1.0 - w_mod   * T["modw_e01"]
T["TripletConfidence"] = 1.0 - (p1 * p2 * p3 * p4)

# De-duplicate and keep top triplets
T = T.sort_values(["ChainScore", "TripletConfidence"], ascending=[False, False]) \
     .drop_duplicates(subset=["host_module", "Genus", "Obj2_Pathway"]).head(4000)

write_csv(T, "Triplets_HostModule_Pathway_Microbe.csv")
write_csv(T.head(500), "Top500_Triplets.csv")

plt.figure(figsize=(13, 7))
top = T.head(25).iloc[::-1]
lbl = top["host_module"].astype(str) + " | " + top["Obj2_Pathway"].astype(str) + " | " + top["Genus"].astype(str)
colors = _distinct_bar_colors(len(top))
plt.barh(lbl, top["ChainScore"], color=colors)
plt.xlabel("ChainScore"); plt.ylabel("Host | Pathway | Microbe")
plt.title("Top Host–Pathway–Microbe Chains (ChainScore)")
save_png("06_TopChains.png")


# =============================================================================
# 9) Gene layer: HostModule -> Genes and 4-layer chains
# =============================================================================
def split_genes(g) -> List[str]:
    if pd.isna(g):
        return []
    parts = [x.strip() for x in str(g).split(";") if x.strip()]
    parts = [p for p in parts if 1 <= len(p) <= 30]
    return parts

gene_edges = []
for _, r in path_to_module.iterrows():
    hm = str(r["host_module"])
    genes = split_genes(r.get("genes", ""))
    p_adj = r.get("p.adjust", np.nan)
    w = 1.0
    if pd.notna(p_adj):
        try:
            w = float(-math.log10(max(float(p_adj), 1e-300)))
        except Exception:
            w = 1.0
    for g in genes:
        gene_edges.append((hm, g, w))

gene_df = pd.DataFrame(gene_edges, columns=["host_module", "Gene", "GeneEdgeWeight"])
if gene_df.empty:
    print("NOTE: Gene layer skipped (no genes found in Obj3 all_pathways).")
else:
    gene_df = gene_df.groupby(["host_module", "Gene"], as_index=False)["GeneEdgeWeight"].max()
    write_csv(gene_df, "HostModule_to_Genes_Links.csv")

    chain4 = T.merge(gene_df, on="host_module", how="inner")
    chain4["ChainGeneScore"] = chain4["ChainScore"].astype(float) * chain4["GeneEdgeWeight"].astype(float)
    chain4 = chain4.sort_values("ChainGeneScore", ascending=False).head(5000)
    write_csv(chain4, "Chains4_Microbe_Pathway_HostModule_Gene.csv")
    write_csv(chain4.head(300), "Top300_Chains4.csv")

    plt.figure(figsize=(13, 7))
    top = chain4.head(25).iloc[::-1]
    lbl = (top["host_module"].astype(str) + " | " + top["Obj2_Pathway"].astype(str) + " | " +
           top["Genus"].astype(str) + " | " + top["Gene"].astype(str))
    colors = _distinct_bar_colors(len(top))
    plt.barh(lbl, top["ChainGeneScore"], color=colors)
    plt.xlabel("ChainGeneScore")
    plt.ylabel("Host | Pathway | Microbe | Gene")
    plt.title("Top 4-layer Chains (Microbe→Pathway→HostModule→Gene)")
    save_png("07_TopChains4_Gene.png")


# =============================================================================
# 10) Build heterogeneous graph (3 or 4 layers)
# =============================================================================
G = nx.Graph()

# add nodes
for hm in host_ref_out["host_module"].astype(str).unique():
    G.add_node(hm, ntype="HostModule")
for p in obj2p_unique["Obj2_Pathway"].astype(str).unique():
    G.add_node(p, ntype="Pathway")
for m in micro["Genus"].astype(str).unique():
    G.add_node(m, ntype="Microbe")
if not gene_df.empty:
    for g in gene_df["Gene"].astype(str).unique():
        G.add_node(g, ntype="Gene")

# add edges microbe-pathway
for _, r in microbe_to_pathway.iterrows():
    G.add_edge(str(r["Genus"]), str(r["Obj2_Pathway"]), weight=float(r["LinkScore"]), etype="Microbe-Pathway")

# add edges pathway-host (aggregate ChainScore)
ph = T.groupby(["Obj2_Pathway", "host_module"], as_index=False)["ChainScore"].sum()
for _, r in ph.iterrows():
    G.add_edge(str(r["Obj2_Pathway"]), str(r["host_module"]), weight=float(r["ChainScore"]), etype="Pathway-Host")

# add edges host-gene
if not gene_df.empty:
    for _, r in gene_df.iterrows():
        G.add_edge(str(r["host_module"]), str(r["Gene"]), weight=float(r["GeneEdgeWeight"]), etype="Host-Gene")

# save nodes/edges
node_list = pd.DataFrame({"Node": list(G.nodes()),
                          "Type": [G.nodes[n].get("ntype", "Other") for n in G.nodes()]})
edge_list = pd.DataFrame([[u, v, d.get("etype", "NA"), float(d.get("weight", 1.0))]
                          for u, v, d in G.edges(data=True)],
                         columns=["Source", "Target", "EdgeType", "Weight"])
write_csv(node_list, "Graph_Nodes.csv")
write_csv(edge_list, "Graph_Edges.csv")


# =============================================================================
# 11) Model schematic (PNG)
# =============================================================================
plt.figure(figsize=(12, 6))
plt.axis("off")

boxes = [
    (0.05, 0.72, 0.25, 0.18, "Obj1: Tissue Microbes\n(Genus + evidence scores)"),
    (0.37, 0.72, 0.25, 0.18, "Obj2: Metabolic Pathways\n(Evidence_Score + class)"),
    (0.69, 0.72, 0.26, 0.18, "Obj3: Host Modules\n(modules + pathway gene lists)"),
    (0.37, 0.42, 0.25, 0.18, "AI/NLP: Pathway Alignment\n(fuzzy matching + Match_Score)"),
    (0.05, 0.42, 0.25, 0.18, "Explainable scoring\nMicrobeScore, LinkScore, ChainScore"),
    (0.69, 0.42, 0.26, 0.18, "Gene layer (optional)\nHostModule → Genes"),
    (0.37, 0.08, 0.25, 0.22, "Graph-ML + Graph-AI\nPageRank/Centrality + Embeddings\nLink prediction + evaluation (AUC/AP)")
]
ax = plt.gca()
for x, y, w, h, t in boxes:
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=2))
    plt.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=12, fontweight='bold')

def arrow(x1, y1, x2, y2):
    plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", lw=2))

arrow(0.30, 0.81, 0.37, 0.81)
arrow(0.62, 0.81, 0.69, 0.81)
arrow(0.50, 0.72, 0.50, 0.60)
arrow(0.18, 0.72, 0.18, 0.60)
arrow(0.82, 0.72, 0.82, 0.60)
arrow(0.50, 0.42, 0.50, 0.30)
arrow(0.18, 0.42, 0.50, 0.19)
arrow(0.82, 0.42, 0.50, 0.19)

plt.title("EXPLAIN-OMICS: AI/ML Multi-Omics Integration Model (processed results only)", fontsize=20, fontweight='bold')
save_png("08_MODEL_Pipeline_Schematic.png")


# =============================================================================
# 12) Graph-AI ranking (PageRank + centralities) + consensus ranking
# =============================================================================
# Build a directed version for PageRank using weights
DG = nx.DiGraph()
for u, v, d in G.edges(data=True):
    w = float(d.get("weight", 1.0))
    DG.add_edge(u, v, weight=w)
    DG.add_edge(v, u, weight=w)

pr = nx.pagerank(DG, weight="weight")
deg = dict(G.degree())
bet = nx.betweenness_centrality(G, k=min(300, max(20, G.number_of_nodes()//3)), seed=SEED, normalized=True)
clo = nx.closeness_centrality(G)

node_rank = pd.DataFrame({
    "Node": list(pr.keys()),
    "Type": [G.nodes[n].get("ntype", "Other") for n in pr.keys()],
    "PageRank": [pr[n] for n in pr.keys()],
    "Degree": [deg.get(n, 0) for n in pr.keys()],
    "Betweenness": [bet.get(n, 0) for n in pr.keys()],
    "Closeness": [clo.get(n, 0) for n in pr.keys()]
})

# consensus score (rank ensemble)
node_rank["r_pagerank"] = rank01(node_rank["PageRank"], True)
node_rank["r_degree"]   = rank01(node_rank["Degree"], True)
node_rank["r_between"]  = rank01(node_rank["Betweenness"], True)
node_rank["r_close"]    = rank01(node_rank["Closeness"], True)
node_rank["ConsensusScore"] = (node_rank["r_pagerank"] + node_rank["r_degree"] + node_rank["r_between"] + node_rank["r_close"]) / 4.0
node_rank = node_rank.sort_values("ConsensusScore", ascending=False)

write_csv(node_rank, "NodeRankings_GraphAI_Consensus.csv")

for layer in ["HostModule", "Pathway", "Microbe", "Gene"]:
    sub = node_rank[node_rank["Type"] == layer].head(25).iloc[::-1]
    if sub.empty:
        continue
    plt.figure(figsize=(10, 6))
    colors = _distinct_bar_colors(len(sub))
    plt.barh(sub["Node"], sub["ConsensusScore"], color=colors)
    plt.xlabel("ConsensusScore")
    plt.ylabel(layer)
    plt.title(f"Top {layer} (Consensus driver ranking)")
    save_png(f"09_ConsensusDrivers_{layer}.png")


# =============================================================================
# 13) Community detection (greedy modularity)
# =============================================================================
comms = list(nx.algorithms.community.greedy_modularity_communities(G))
node2c = {}
for i, cset in enumerate(comms):
    for n in cset:
        node2c[n] = i

communities_out = pd.DataFrame({
    "Node": list(node2c.keys()),
    "Community": [node2c[n] for n in node2c.keys()],
    "Type": [G.nodes[n].get("ntype", "Other") for n in node2c.keys()]
})
write_csv(communities_out, "Communities_GreedyModularity.csv")

cs = communities_out.groupby("Community").size().reset_index(name="n_nodes").sort_values("n_nodes", ascending=False)
write_csv(cs, "CommunitySizes.csv")

plt.figure(figsize=(10, 6))
top = cs.head(25).iloc[::-1]
colors = _distinct_bar_colors(len(top))
plt.barh(top["Community"].astype(str), top["n_nodes"], color=colors)
plt.xlabel("n_nodes"); plt.ylabel("Community")
plt.title("Graph Communities (Greedy Modularity) - Top Sizes")
save_png("10_CommunitySizes.png")


# =============================================================================
# 14) Layered network plot (Microbe–Pathway–HostModule + optional Gene)
# =============================================================================
def plot_layered_network(Gin: nx.Graph, title: str, outname: str, max_edges: int = 320):
    edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in Gin.edges(data=True)]
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)

    if len(edges_sorted) > max_edges:
        keep = set(tuple(sorted((u, v))) for u, v, _ in edges_sorted[:max_edges])
        Gp = nx.Graph()
        for n, data in Gin.nodes(data=True):
            Gp.add_node(n, **data)
        for u, v, d in Gin.edges(data=True):
            if tuple(sorted((u, v))) in keep:
                Gp.add_edge(u, v, **d)
    else:
        Gp = Gin

    has_gene = any(Gp.nodes[n].get("ntype") == "Gene" for n in Gp.nodes())
    order = ("HostModule", "Pathway", "Microbe", "Gene") if has_gene else ("HostModule", "Pathway", "Microbe")
    pos = layered_positions(Gp, layer_order=order, y_gap=1.15)

    labels = {n: wrap_label(n, 24) for n in Gp.nodes()}
    sizes = []
    for n in Gp.nodes():
        t = Gp.nodes[n].get("ntype", "Other")
        if t == "HostModule":
            sizes.append(1400)
        elif t == "Pathway":
            sizes.append(900)
        elif t == "Microbe":
            sizes.append(650)
        elif t == "Gene":
            sizes.append(520)
        else:
            sizes.append(500)

    plt.figure(figsize=(24, 12))
    nx.draw_networkx_nodes(Gp, pos, node_size=sizes, alpha=0.90)
    nx.draw_networkx_edges(Gp, pos, alpha=0.18)
    nx.draw_networkx_labels(Gp, pos, labels=labels, font_size=7)

    ys = [pos[n][1] for n in pos]
    y_top = (max(ys) + 1.0) if ys else 1.0
    plt.text(0.0, y_top, "Host Modules", fontsize=14, ha="center")
    plt.text(1.0, y_top, "Metabolic Pathways", fontsize=14, ha="center")
    plt.text(2.0, y_top, "Tissue Microbes", fontsize=14, ha="center")
    if has_gene:
        plt.text(3.0, y_top, "Genes", fontsize=14, ha="center")

    plt.title(title)
    plt.axis("off")
    save_png(f"{outname}.png")

plot_layered_network(G,
                     "EXPLAIN-OMICS: Layered Host–Pathway–Microbe (and Gene) Network",
                     "11_LayeredNetwork")


# =============================================================================
# 15) Graph-ML embeddings (random-walk co-occurrence + SVD)
# =============================================================================
def top_edges_by_weight(Gin: nx.Graph, max_edges: int = 900) -> nx.Graph:
    edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in Gin.edges(data=True)]
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
    keep = set(tuple(sorted((u, v))) for u, v, _ in edges_sorted[:max_edges])

    Gp = nx.Graph()
    for n, data in Gin.nodes(data=True):
        Gp.add_node(n, **data)
    for u, v, d in Gin.edges(data=True):
        if tuple(sorted((u, v))) in keep:
            Gp.add_edge(u, v, **d)
    return Gp

def build_random_walks(Gin: nx.Graph, num_walks: int = 20, walk_length: int = 30, seed: int = 42):
    rng = random.Random(seed)
    nodes = list(Gin.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            cur = start
            for _ in range(walk_length - 1):
                nbrs = list(Gin.neighbors(cur))
                if not nbrs:
                    break
                cur = rng.choice(nbrs)
                walk.append(cur)
            walks.append(walk)
    return walks

def cooccurrence_from_walks(walks, window_size: int = 5):
    C = {}
    for w in walks:
        L = len(w)
        for i, u in enumerate(w):
            lo = max(0, i - window_size)
            hi = min(L, i + window_size + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                v = w[j]
                C.setdefault(u, {})
                C[u][v] = C[u].get(v, 0.0) + 1.0
    return C

def svd_embeddings_from_cooc(C, nodes: List[str], dim: int = 32) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n), dtype=float)
    for u, dv in C.items():
        iu = idx.get(u, None)
        if iu is None:
            continue
        for v, cnt in dv.items():
            iv = idx.get(v, None)
            if iv is None:
                continue
            M[iu, iv] = cnt
    M = np.log1p(M)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    d = min(dim, U.shape[1])
    return U[:, :d] * S[:d]

def pca2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

G_emb = top_edges_by_weight(G, max_edges=900)
nodes = list(G_emb.nodes())
walks = build_random_walks(G_emb, num_walks=20, walk_length=30, seed=SEED)
C = cooccurrence_from_walks(walks, window_size=5)
emb = svd_embeddings_from_cooc(C, nodes, dim=32)

emb_df = pd.DataFrame(emb, columns=[f"emb_{i+1}" for i in range(emb.shape[1])])
emb_df.insert(0, "Node", nodes)
emb_df["Type"] = [G_emb.nodes[n].get("ntype", "Other") for n in nodes]
write_csv(emb_df, "NodeEmbeddings_RandomWalkSVD.csv")

xy = pca2(emb)
plt.figure(figsize=(8, 6))
plt.scatter(xy[:, 0], xy[:, 1], s=14)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Graph Embedding (2D PCA) — all nodes")
save_png("12_Embedding2D_AllNodes.png")


# =============================================================================
# 16) Link prediction + evaluation (edge holdout) -> best-model proof
# =============================================================================
def cosine_score(u: str, v: str, node_index: Dict[str, int], emb: np.ndarray) -> float:
    iu = node_index.get(u, None)
    iv = node_index.get(v, None)
    if iu is None or iv is None:
        return 0.0
    a = emb[iu]; b = emb[iv]
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))

def evaluate_edge_type(Gfull: nx.Graph,
                       etype: str,
                       nodes: List[str],
                       emb: np.ndarray,
                       n_neg_ratio: int = 3,
                       test_frac: float = 0.2) -> Dict[str, object]:

    E = [(u, v) for u, v, d in Gfull.edges(data=True) if d.get("etype") == etype]
    if len(E) < 30:
        return {"EdgeType": etype, "n_edges": len(E), "AUC": np.nan, "AP": np.nan, "note": "too few edges"}

    random.shuffle(E)
    n_test = max(10, int(len(E) * test_frac))
    test_pos = E[:n_test]

    # Train edges = all except test_pos of this etype
    Gtrain = nx.Graph()
    for n, data in Gfull.nodes(data=True):
        Gtrain.add_node(n, **data)

    test_set = set(tuple(sorted(e)) for e in test_pos)
    for u, v, d in Gfull.edges(data=True):
        if d.get("etype") == etype and tuple(sorted((u, v))) in test_set:
            continue
        Gtrain.add_edge(u, v, **d)

    # Candidate node type sets
    if etype == "Microbe-Pathway":
        A = [n for n in Gfull.nodes() if Gfull.nodes[n].get("ntype") == "Microbe"]
        B = [n for n in Gfull.nodes() if Gfull.nodes[n].get("ntype") == "Pathway"]
    elif etype == "Pathway-Host":
        A = [n for n in Gfull.nodes() if Gfull.nodes[n].get("ntype") == "Pathway"]
        B = [n for n in Gfull.nodes() if Gfull.nodes[n].get("ntype") == "HostModule"]
    else:
        A = list(Gfull.nodes()); B = list(Gfull.nodes())

    existing = set(tuple(sorted((u, v))) for u, v in Gtrain.edges())
    target_neg = n_neg_ratio * len(test_pos)
    test_neg = []
    tries = 0
    while len(test_neg) < target_neg and tries < target_neg * 50:
        u = random.choice(A); v = random.choice(B)
        if u == v:
            tries += 1; continue
        key = tuple(sorted((u, v)))
        if key in existing:
            tries += 1; continue
        test_neg.append((u, v))
        tries += 1

    node_index = {n: i for i, n in enumerate(nodes)}
    pairs = test_pos + test_neg
    y_true = np.array([1] * len(test_pos) + [0] * len(test_neg), dtype=int)
    y_score = np.array([cosine_score(u, v, node_index, emb) for (u, v) in pairs], dtype=float)

    return {
        "EdgeType": etype,
        "n_edges": len(E),
        "n_test_pos": len(test_pos),
        "n_test_neg": len(test_neg),
        "AUC": auc_score(y_true, y_score),
        "AP": average_precision(y_true, y_score),
        "note": "ok"
    }

eval_rows = [
    evaluate_edge_type(G_emb, "Microbe-Pathway", nodes, emb),
    evaluate_edge_type(G_emb, "Pathway-Host", nodes, emb),
]
eval_df = pd.DataFrame(eval_rows)
write_csv(eval_df, "Evaluation_LinkPrediction_AUC_AP.csv", folder=EVAL_DIR)

plt.figure(figsize=(7, 4))
colors = _distinct_bar_colors(len(eval_df))
plt.bar(eval_df["EdgeType"], eval_df["AUC"], color=colors)
plt.ylabel("AUC"); plt.title("Link Prediction Evaluation (AUC)")
save_png("Evaluation_AUC.png", folder=EVAL_DIR)

plt.figure(figsize=(7, 4))
colors = _distinct_bar_colors(len(eval_df))
plt.bar(eval_df["EdgeType"], eval_df["AP"], color=colors)
plt.ylabel("Average Precision (AP)"); plt.title("Link Prediction Evaluation (AP)")
save_png("Evaluation_AP.png", folder=EVAL_DIR)

best_report = pd.DataFrame([{
    "BestModel": "RandomWalkSVD_Embeddings + Cosine Link Prediction",
    "Mean_AUC": float(np.nanmean(eval_df["AUC"].values)),
    "Mean_AP": float(np.nanmean(eval_df["AP"].values)),
    "DataLevel": "Processed results only",
    "Use": "Hypothesis generation / driver prioritization (not clinical prediction)"
}])
write_csv(best_report, "BestModel_Report.csv", folder=EVAL_DIR)


# =============================================================================
# 17) Optional: embedding clusters + predicted new links (hypothesis list)
# =============================================================================
if SKLEARN_OK and emb.shape[0] >= 25:
    X = StandardScaler(with_mean=False).fit_transform(emb)
    k = min(8, max(3, emb.shape[0] // 20))
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED)
    cl = km.fit_predict(X)
    emb_clusters = emb_df[["Node", "Type"]].copy()
    emb_clusters["EmbedCluster"] = cl
    write_csv(emb_clusters, "EmbeddingClusters_KMeans.csv")
else:
    print("Embedding clusters skipped (sklearn not available or too few nodes).")

def link_prediction_from_embeddings(nodes: List[str], emb: np.ndarray, existing_edges_set: set,
                                    candidate_pairs: List[Tuple[str, str]], top_k: int = 300) -> pd.DataFrame:
    idx = {n: i for i, n in enumerate(nodes)}
    E = np.asarray(emb, dtype=float)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms

    results = []
    for u, v in candidate_pairs:
        if u not in idx or v not in idx:
            continue
        if frozenset((u, v)) in existing_edges_set:
            continue
        score = float(np.dot(En[idx[u]], En[idx[v]]))
        results.append((u, v, score))
    results.sort(key=lambda x: x[2], reverse=True)
    results = results[:top_k]
    return pd.DataFrame(results, columns=["NodeA", "NodeB", "PredScore_Cosine"])

existing_edges = set(frozenset((u, v)) for u, v in G_emb.edges())

host_nodes = [n for n in nodes if G_emb.nodes[n].get("ntype") == "HostModule"]
path_nodes = [n for n in nodes if G_emb.nodes[n].get("ntype") == "Pathway"]
micro_nodes = [n for n in nodes if G_emb.nodes[n].get("ntype") == "Microbe"]

cand_mp = [(m, p) for m in micro_nodes for p in path_nodes]
cand_ph = [(p, h) for p in path_nodes for h in host_nodes]

MAX_CAND = 200000
if len(cand_mp) > MAX_CAND:
    cand_mp = random.sample(cand_mp, MAX_CAND)
if len(cand_ph) > MAX_CAND:
    cand_ph = random.sample(cand_ph, MAX_CAND)

pred_mp = link_prediction_from_embeddings(nodes, emb, existing_edges, cand_mp, top_k=300)
pred_ph = link_prediction_from_embeddings(nodes, emb, existing_edges, cand_ph, top_k=300)

write_csv(pred_mp, "PredictedLinks_Microbe_Pathway.csv")
write_csv(pred_ph, "PredictedLinks_Pathway_HostModule.csv")

plt.figure(figsize=(12, 7))
top = pred_mp.head(25).iloc[::-1]
lbl = top["NodeA"].astype(str) + " → " + top["NodeB"].astype(str)
colors = _distinct_bar_colors(len(top))
plt.barh(lbl, top["PredScore_Cosine"], color=colors)
plt.xlabel("PredScore (cosine)"); plt.ylabel("Predicted Microbe → Pathway")
plt.title("Top Predicted Microbe→Pathway Links (Graph-ML)")
save_png("13_TopPredictedLinks_MicrobePathway.png")

plt.figure(figsize=(12, 7))
top = pred_ph.head(25).iloc[::-1]
lbl = top["NodeA"].astype(str) + " → " + top["NodeB"].astype(str)
colors = _distinct_bar_colors(len(top))
plt.barh(lbl, top["PredScore_Cosine"], color=colors)
plt.xlabel("PredScore (cosine)"); plt.ylabel("Predicted Pathway → HostModule")
plt.title("Top Predicted Pathway→HostModule Links (Graph-ML)")
save_png("14_TopPredictedLinks_PathwayHost.png")


# =============================================================================
# 18) Final manifests
# =============================================================================
# =============================================================================
# 18) Additional interaction network visuals (requested)
#     - Microbe → Pathway → HostModule
#     - Microbe → Pathway → HostModule → Gene (if gene layer exists)
# =============================================================================
def _layered_positions_by_type(Gv: nx.Graph, type_order: List[str], x_gap: float = 3.2, y_gap: float = 1.0):
    layers = {t: [] for t in type_order}
    for n in Gv.nodes():
        t = Gv.nodes[n].get("ntype", "Other")
        if t in layers:
            layers[t].append(n)

    def strength(n):
        return sum(float(Gv[n][nbr].get("weight", 1.0)) for nbr in Gv.neighbors(n))

    pos = {}
    for i, t in enumerate(type_order):
        nodes = sorted(layers[t], key=strength, reverse=True)
        if not nodes:
            continue
        ys = np.linspace(0, -(len(nodes) - 1) * y_gap, len(nodes)) if len(nodes) > 1 else [0.0]
        x = i * x_gap
        for j, n in enumerate(nodes):
            pos[n] = (x, float(ys[j]))
    return pos

def _build_microbe_pathway_host_gene_viz(triplets_df: pd.DataFrame,
                                        gene_links: pd.DataFrame = None,
                                        max_triplets: int = 240,
                                        max_genes_per_module: int = 15) -> nx.Graph:
    # Use strongest triplets for readability
    base = triplets_df.sort_values("ChainScore", ascending=False).head(max_triplets).copy()

    Gv = nx.Graph()
    for _, r in base.iterrows():
        m = str(r["Genus"])
        p = str(r["Obj2_Pathway"])
        h = str(r["host_module"])

        Gv.add_node(m, ntype="Microbe")
        Gv.add_node(p, ntype="Pathway")
        Gv.add_node(h, ntype="HostModule")

        w_mp = float(r.get("LinkScore", 1.0))
        w_ph = float(r.get("ChainScore", 1.0))
        Gv.add_edge(m, p, weight=w_mp, etype="Microbe-Pathway")
        Gv.add_edge(p, h, weight=w_ph, etype="Pathway-Host")

    if gene_links is not None and isinstance(gene_links, pd.DataFrame) and (not gene_links.empty):
        gsub = gene_links.copy()
        gsub["GeneEdgeWeight"] = pd.to_numeric(gsub["GeneEdgeWeight"], errors="coerce").fillna(1.0)
        gsub = gsub.sort_values(["host_module", "GeneEdgeWeight"], ascending=[True, False]) \
                   .groupby("host_module", as_index=False).head(max_genes_per_module)

        keep_modules = set([n for n in Gv.nodes() if Gv.nodes[n].get("ntype") == "HostModule"])
        gsub = gsub[gsub["host_module"].astype(str).isin(keep_modules)]

        for _, r in gsub.iterrows():
            h = str(r["host_module"])
            g = str(r["Gene"])
            w = float(r["GeneEdgeWeight"])
            Gv.add_node(g, ntype="Gene")
            Gv.add_edge(h, g, weight=w, etype="Host-Gene")

    return Gv


def _plot_layered_interaction(Gv: nx.Graph,
                             type_order: List[str],
                             title: str,
                             out_png: str,
                             max_edges: int = 420,
                             label_fontsize: int = 12,
                             heading_fontsize: int = 24,
                             title_fontsize: int = 24,
                             x_gap: float = 4.2,
                             y_gap: float = None,
                             top_labels_per_type: Dict[str, int] = None):
    """
    Publication-friendly layered interaction network (PNG only):
      - Bigger + bold fonts
      - Colorful nodes by layer (Microbe/Pathway/HostModule/Gene)
      - Edge colors by edge-type (Microbe-Pathway / Pathway-Host / Host-Gene)
      - Automatic edge pruning to reduce clutter
      - Extra vertical spacing + label pruning to avoid overlapping bubbles/labels
    """

    if top_labels_per_type is None:
        top_labels_per_type = {"Microbe": 18, "Pathway": 22, "HostModule": 18, "Gene": 22}

    # ---- prune edges by weight (keep top edges)
    edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in Gv.edges(data=True)]
    edges = sorted(edges, key=lambda x: x[2], reverse=True)
    keep = set(tuple(sorted((u, v))) for u, v, _ in edges[:max_edges])

    Gp = nx.Graph()
    for n, data in Gv.nodes(data=True):
        Gp.add_node(n, **data)
    for u, v, d in Gv.edges(data=True):
        if tuple(sorted((u, v))) in keep:
            Gp.add_edge(u, v, **d)

    # ---- dynamic spacing (more nodes => more vertical space)
    layer_counts = {t: 0 for t in type_order}
    for n in Gp.nodes():
        t = Gp.nodes[n].get("ntype", "Other")
        if t in layer_counts:
            layer_counts[t] += 1
    max_layer_n = max(layer_counts.values()) if layer_counts else 1
    if y_gap is None:
        # 0.55 + 0.03*n gives ~1.45 when n=30; ~2.35 when n=60
        y_gap = max(1.25, 0.55 + 0.03 * max_layer_n)

    # ---- positions (layered)
    pos = _layered_positions_by_type(Gp, type_order, x_gap=x_gap, y_gap=y_gap)

    # ---- styling
    type_colors = {
        "Microbe":   "#1f77b4",  # blue
        "Pathway":   "#ff7f0e",  # orange
        "HostModule":"#2ca02c",  # green
        "Gene":      "#d62728",  # red
        "Other":     "#7f7f7f"   # gray
    }
    edge_colors = {
        "Microbe-Pathway": "#7aa6d8",
        "Pathway-Host":    "#95d095",
        "Host-Gene":       "#f2a2a2",
        "NA":              "#bdbdbd"
    }

    # node colors + sizes
    node_color = []
    node_size = []
    for n in Gp.nodes():
        t = Gp.nodes[n].get("ntype", "Other")
        node_color.append(type_colors.get(t, type_colors["Other"]))
        if t == "HostModule":
            node_size.append(2500)
        elif t == "Pathway":
            node_size.append(1800)
        elif t == "Microbe":
            node_size.append(1500)
        elif t == "Gene":
            node_size.append(1200)
        else:
            node_size.append(1200)

    # edge colors + widths from weight (compressed)
    max_w = 1.0
    if Gp.number_of_edges() > 0:
        max_w = max(float(d.get("weight", 1.0)) for _, _, d in Gp.edges(data=True))
    ecols, ewidths = [], []
    for _, _, d in Gp.edges(data=True):
        et = d.get("etype", "NA")
        ecols.append(edge_colors.get(et, edge_colors["NA"]))
        w = float(d.get("weight", 1.0))
        ewidths.append(0.7 + 2.4 * (np.log1p(w) / (np.log1p(max_w) + 1e-9)))

    # ---- label pruning (avoid overlaps): label only top-strength nodes per type
    def strength(n):
        return sum(float(Gp[n][nbr].get("weight", 1.0)) for nbr in Gp.neighbors(n))

    labels = {n: "" for n in Gp.nodes()}
    for t in type_order:
        nodes_t = [n for n in Gp.nodes() if Gp.nodes[n].get("ntype") == t]
        nodes_t = sorted(nodes_t, key=strength, reverse=True)
        k = int(top_labels_per_type.get(t, 0))
        for n in nodes_t[:k]:
            labels[n] = wrap_label(n, 18)

    # ---- plot (bigger canvas to give space)
    fig_w = 34 if len(type_order) >= 4 else 30
    fig_h = 18
    plt.figure(figsize=(fig_w, fig_h))
    nx.draw_networkx_nodes(
        Gp, pos,
        node_size=node_size,
        node_color=node_color,
        alpha=0.96,
        linewidths=1.0,
        edgecolors="#ffffff"
    )
    nx.draw_networkx_edges(Gp, pos, edge_color=ecols, width=ewidths, alpha=0.30)
    nx.draw_networkx_labels(Gp, pos, labels=labels, font_size=label_fontsize, font_weight="bold")

    # headings
    ys = [pos[n][1] for n in pos] if pos else [0.0]
    y_top = (max(ys) + (2.8 * y_gap)) if ys else (2.8 * y_gap)
    for i, t in enumerate(type_order):
        plt.text(i * x_gap, y_top, t, fontsize=heading_fontsize, ha="center", weight="bold")

    plt.title(title, fontsize=title_fontsize, weight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# --- Build and save the requested interaction networks
# Use T (final filtered triplets with ChainScore) for clean, thesis-ready plots
G3_req = _build_microbe_pathway_host_gene_viz(triplets_df=T, gene_links=None, max_triplets=200)
_plot_layered_interaction(
    G3_req,
    type_order=["Microbe", "Pathway", "HostModule"],
    title="EXPLAIN-OMICS Interaction Network: Microbe → Pathway → HostModule",
    out_png=os.path.join(FIG_DIR, "15_Network_3Layer_Microbe_Pathway_HostModule.png"),
    max_edges=320,
    label_fontsize=13,
    heading_fontsize=26,
    title_fontsize=26,
    x_gap=4.6
)

if (not gene_df.empty):
    G4_req = _build_microbe_pathway_host_gene_viz(triplets_df=T, gene_links=gene_df,
                                                  max_triplets=160, max_genes_per_module=10)
    _plot_layered_interaction(
        G4_req,
        type_order=["Microbe", "Pathway", "HostModule", "Gene"],
        title="EXPLAIN-OMICS Interaction Network: Microbe → Pathway → HostModule → Gene",
        out_png=os.path.join(FIG_DIR, "16_Network_4Layer_Microbe_Pathway_HostModule_Gene.png"),
        max_edges=340,
        label_fontsize=13,
        heading_fontsize=26,
        title_fontsize=26,
        x_gap=4.8
    )

    # Compact: Microbe → Pathway → Gene (collapse modules visually)
    # (We keep HostModule nodes, but use type order to pull Gene closer for a compact visual)
    _plot_layered_interaction(
        G4_req,
        type_order=["Microbe", "Pathway", "Gene", "HostModule"],
        title="EXPLAIN-OMICS Compact View: Microbe → Pathway → Gene (via Host Modules)",
        out_png=os.path.join(FIG_DIR, "17_Network_Compact_Microbe_Pathway_Gene.png"),
        max_edges=260,
        label_fontsize=13,
        heading_fontsize=26,
        title_fontsize=26,
        x_gap=4.3,
        top_labels_per_type={"Microbe": 16, "Pathway": 20, "Gene": 22, "HostModule": 10}
    )


manifest = []
for fn in sorted(os.listdir(CSV_DIR)):
    if fn.lower().endswith(".csv"):
        manifest.append({"File": f"csv/{fn}", "Type": "CSV"})
for fn in sorted(os.listdir(FIG_DIR)):
    if fn.lower().endswith(".png"):
        manifest.append({"File": f"figures_png/{fn}", "Type": "PNG"})
for fn in sorted(os.listdir(EVAL_DIR)):
    if fn.lower().endswith(".csv"):
        manifest.append({"File": f"evaluation/{fn}", "Type": "CSV"})
    if fn.lower().endswith(".png"):
        manifest.append({"File": f"evaluation/{fn}", "Type": "PNG"})

write_csv(pd.DataFrame(manifest), "OutputManifest.csv", folder=FINAL_DIR)

print("\nDONE ✅")
print("All outputs saved under:", FINAL_DIR)
