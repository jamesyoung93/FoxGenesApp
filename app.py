# FOX-Gene Complement Explorer – v5 (Merged & Improved)
# Combines best features from v2 (conservation filters, GitHub links)
# and v4 (ortholog panel viewer, model selection)
# Plus: gene search, summary stats, cleaner UI, better documentation

import io
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib_venn import venn2
from wordcloud import WordCloud, STOPWORDS

##############################################################################
# Configuration
##############################################################################

APP_VERSION = "5.0"

DEFAULT_FILES = [
    "FOX_unknown_app_table_cleaned.csv",
    "FOX_unknown_app_table_UPDATED_with_panel_protein_ids.csv",
    "FOX_unknown_app_table_UPDATED.csv",
    "FOX_unknown_with_hits_function_greedy_enriched.csv",
]

WC_SEED = 42
EXTRA_STOP = {
    "protein", "putative", "family", "domain", "predicted", "superfamily",
    "probable", "possible", "like", "related", "ec", "ribosomal",
    "rna", "binding", "subunit"
}
STOPWORDS_FULL = STOPWORDS.union(EXTRA_STOP)

# Display column ordering preference
DISPLAY_COLS_PRIORITY = [
    "rank_unknown", "Annotation", "locus_tag", "gene_symbol", "product",
    "Protein_names", "FOX probability", "Gene length", "Prob_per_len",
    "filamentous_diazotroph_hits", "non_diazotroph_hits", "Cyanothece_%ID",
    "diazo_hit_count", "nondiazo_hit_count"
]

# Organism panel mapping for cleaner display
ORGANISM_DISPLAY_NAMES = {
    "anabaena_variabilis_atcc29413": "Anabaena variabilis ATCC 29413",
    "arthrospira_maxima_cs-328": "Arthrospira maxima CS-328",
    "crocosphaera_watsonii_wh8501": "Crocosphaera watsonii WH8501",
    "cyanothece_sp_atcc51142": "Cyanothece sp. ATCC 51142",
    "gloeothece_sp_6803": "Gloeothece sp. 6803",
    "nostoc_azollae_0708": "Nostoc azollae 0708",
    "nostoc_punctiforme_pcc73102": "Nostoc punctiforme PCC 73102",
    "synechococcus_elongatus_pcc7942": "Synechococcus elongatus PCC 7942",
}

##############################################################################
# Helper Functions
##############################################################################

@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def first_existing_file() -> Optional[Path]:
    """Find first available data file from defaults."""
    here = Path(__file__).resolve().parent
    for fn in DEFAULT_FILES:
        p = here / fn
        if p.exists():
            return p
    return None


def collapse_name(name: str) -> str:
    """Collapse protein names for word cloud visualization."""
    if pd.isna(name) or str(name).strip() == "":
        return "Unknown"
    s = str(name).strip()
    low = s.lower()
    if re.match(r"^(all|alr|asl|asr)\d+", s, re.I):
        return "Uncharacterized"
    if "ribosom" in low:
        return "Ribosomal"
    if "hypothetical" in low:
        return "Uncharacterized"
    if any(x in low for x in ["cab", "elip", "hlip"]):
        return "Protective"
    return s


def make_wordcloud(series: pd.Series, title: str, overall: set):
    """Generate word cloud from protein names."""
    collapsed = series.dropna().apply(collapse_name)
    uniq = set(collapsed.unique())
    if "Unknown" in overall and "Unknown" not in uniq:
        uniq.add("Unknown")
    if not uniq:
        st.write(f"*(no names in {title})*")
        return
    txt = " ".join(sorted(uniq))
    wc = WordCloud(
        width=900, height=320, background_color="white",
        stopwords=STOPWORDS_FULL, random_state=WC_SEED,
        colormap="viridis"
    ).generate(txt)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def cumulative_select(df: pd.DataFrame, sort_col: str, length_col: str, 
                      nt_lim: int, dedupe_col: Optional[str] = None) -> pd.DataFrame:
    """Select genes cumulatively until length budget is exhausted."""
    seen = set()
    sel, cum = [], 0
    for _, row in df.iterrows():
        length = row.get(length_col, None)
        if pd.isna(length):
            continue
        # Deduplicate by annotation if requested
        if dedupe_col and dedupe_col in row.index:
            key = row[dedupe_col]
            if key in seen:
                continue
            seen.add(key)
        if cum + float(length) > nt_lim:
            break
        cum += float(length)
        sel.append(row)
    return pd.DataFrame(sel)


def enforce_col_order(tbl: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns for better readability."""
    if tbl.empty:
        return tbl
    cols = list(tbl.columns)
    ordered = []
    for c in DISPLAY_COLS_PRIORITY:
        if c in cols:
            ordered.append(c)
            cols.remove(c)
    # Add remaining columns (exclude wide ortholog columns for cleaner display)
    for c in cols:
        if not any(x in c for x in ["__Prot", "__pident", "__evalue"]):
            ordered.append(c)
    return tbl[ordered]


def download_csv(df: pd.DataFrame, label: str):
    """Create download button for CSV."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(
        f"📥 Download {label}",
        buf.getvalue(),
        file_name=f"{label}.csv",
        mime="text/csv",
    )


def extract_ortholog_panel(row: pd.Series) -> pd.DataFrame:
    """Extract per-organism ortholog information from a gene row."""
    rows = []
    # Check for __Prot alias columns first
    prot_cols = [c for c in row.index if c.endswith("__Prot")]
    if prot_cols:
        for pc in sorted(prot_cols):
            org_key = pc.replace("__Prot", "")
            org_display = ORGANISM_DISPLAY_NAMES.get(org_key, org_key.replace("_", " ").title())
            pid = row.get(f"{org_key}__pident", None)
            ev = row.get(f"{org_key}__evalue", None)
            hit_id = row.get(pc, None)
            if pd.notna(hit_id) and str(hit_id).strip():
                rows.append({
                    "Organism": org_display,
                    "Best Hit ID": hit_id,
                    "% Identity": f"{pid:.1f}" if pd.notna(pid) else "",
                    "E-value": f"{ev:.2e}" if pd.notna(ev) else ""
                })
    
    # Fallback to s_id__* columns
    if not rows:
        raw_prot = [c for c in row.index if c.startswith("s_id__")]
        for pc in sorted(raw_prot):
            org_key = pc.replace("s_id__", "")
            org_display = ORGANISM_DISPLAY_NAMES.get(org_key, org_key.replace("_", " ").title())
            pid = row.get(f"pident__{org_key}", None)
            ev = row.get(f"evalue__{org_key}", None)
            hit_id = row.get(pc, None)
            if pd.notna(hit_id) and str(hit_id).strip():
                rows.append({
                    "Organism": org_display,
                    "Best Hit ID": hit_id,
                    "% Identity": f"{pid:.1f}" if pd.notna(pid) else "",
                    "E-value": f"{ev:.2e}" if pd.notna(ev) else ""
                })
    
    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame, prob_col: str) -> dict:
    """Compute summary statistics for the filtered dataset."""
    stats = {
        "Total genes": len(df),
        "Mean pFOX": df[prob_col].mean() if prob_col in df.columns else None,
        "Median pFOX": df[prob_col].median() if prob_col in df.columns else None,
        "High confidence (pFOX ≥ 0.7)": len(df[df[prob_col] >= 0.7]) if prob_col in df.columns else None,
    }
    if "Gene length" in df.columns:
        stats["Total coding sequence"] = f"{int(df['Gene length'].sum()):,} nt"
    return stats


##############################################################################
# Main App
##############################################################################

st.set_page_config(
    page_title="FOX-Gene Complement Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .block-container { padding-top: 2rem; }
    div[data-testid="stExpander"] details summary p { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Title and header
st.title("🧬 FOX-Gene Complement Explorer")
st.caption(f"v{APP_VERSION} — Design FOX-gene complements for synthetic nitrogen fixation")

# Brief explanation
st.markdown("""
This tool supports the design of FOX gene complements for transferring oxic nitrogen fixation 
capability to non-diazotrophic hosts. It uses machine learning predictions trained on 
*Anabaena* sp. PCC 7120 multi-omic data to rank candidate FOX (Fixation in the presence of OXygen) 
genes and applies comparative bioinformatics filters based on conservation across diazotrophic 
and non-diazotrophic cyanobacteria.
""")

# GitHub/resource links banner (restored from v2)
with st.expander("📚 Resources & Source Code", expanded=False):
    st.markdown(
        """
        • **Cohort-BLAST workflow** → [cyanobacteria-diazotrophic-proteome](https://github.com/jamesyoung93/cyanobacteria-diazotrophic-proteome)  
        • **Streamlit app source** → [FoxGenesApp](https://github.com/jamesyoung93/FoxGenesApp)  
        • **ML feature-engineering / modeling** → [FoxGenes_ML](https://github.com/jamesyoung93/FoxGenes_ML)
        
        **Supplementary Materials**: Tables S1-S7 provide complete gene lists, feature importance, 
        proteome accessions, filter definitions, and model performance metrics.
        """,
        unsafe_allow_html=False,
    )

st.divider()

# Load data
data_path = first_existing_file()
if data_path is None:
    st.warning("📂 Upload an app-ready CSV to begin")
    up = st.file_uploader("Upload FOX gene data CSV", ["csv"])
    if up is None:
        st.info("Expected columns include: Annotation, FOX probability, Gene length, Prob_per_len, "
                "filamentous_diazotroph_hits, non_diazotroph_hits, Cyanothece_%ID")
        st.stop()
    df = pd.read_csv(up)
else:
    df = read_csv(str(data_path))
    st.sidebar.success(f"📄 Loaded: {data_path.name}")

# Normalize column names
if "ENS_PRED" in df.columns and "FOX probability" not in df.columns:
    df.rename(columns={"ENS_PRED": "FOX probability"}, inplace=True)

# Identify available probability columns (with display names)
PROB_COL_LABELS = {
    "FOX probability": "Ensemble with position (default)",
    "prob_ensemble_mean_no_position": "Ensemble without position",
    "prob_ensemble_mean_with_position": "Ensemble with position",
    "prob_logreg_with_position": "Logistic Regression (with pos)",
    "prob_rf_with_position": "Random Forest (with pos)",
    "prob_xgb_with_position": "XGBoost (with pos)",
    "prob_logreg_no_position": "Logistic Regression (no pos)",
    "prob_rf_no_position": "Random Forest (no pos)",
    "prob_xgb_no_position": "XGBoost (no pos)",
}

prob_cols = [c for c in [
    "FOX probability",
    "prob_ensemble_mean_no_position",
    "prob_logreg_with_position",
    "prob_rf_with_position", 
    "prob_xgb_with_position",
    "prob_logreg_no_position",
    "prob_rf_no_position",
    "prob_xgb_no_position",
] if c in df.columns]

if not prob_cols:
    st.error("❌ No probability column found. Expected 'FOX probability' or ensemble probability columns.")
    st.stop()

# Coerce numeric columns
num_cols = ["FOX probability", "Gene length", "Prob_per_len", "Prob_per_kb",
            "filamentous_diazotroph_hits", "non_diazotroph_hits", 
            "Cyanothece_%ID", "Croco_%ID",  # support both column names
            "diazo_hit_count", "nondiazo_hit_count"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Derive conservation flags (restored from v2)
if "filamentous_diazotroph_hits" in df.columns:
    df["Filamentous_cons"] = df["filamentous_diazotroph_hits"].fillna(0).gt(0).map(
        {True: "Conserved", False: "Not conserved"}
    )
else:
    df["Filamentous_cons"] = "Unknown"

if "non_diazotroph_hits" in df.columns:
    nd_hit = df["non_diazotroph_hits"].fillna(0).gt(0)
    df["ND_cons"] = nd_hit.map({True: "Hit", False: "No hit"})
else:
    df["ND_cons"] = "Unknown"

# Cyanothece hit (support both old Croco_ and new Cyanothece_ column names)
cyano_col = "Cyanothece_%ID" if "Cyanothece_%ID" in df.columns else "Croco_%ID"
if cyano_col in df.columns:
    df["Cyano_hit"] = df[cyano_col].fillna(0).ge(80).map({True: "Yes", False: "No"})
else:
    df["Cyano_hit"] = "Unknown"

##############################################################################
# Sidebar Filters
##############################################################################

with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model selection
    st.markdown("**Model Selection**")
    
    # Add model explanation
    with st.expander("ℹ️ About the ML models", expanded=False):
        st.markdown("""
        **Ensemble (default)**: Mean of Logistic Regression, Random Forest, and XGBoost 
        predictions. Recommended for most use cases.
        
        **With position**: Models trained including `chromosome_region_start/end` as features. 
        May capture genomic clustering of FOX genes but could overfit to well-studied islands.
        
        **Without position**: Ablation models excluding genomic position features. May better 
        generalize to genes in less-studied regions (e.g., *hupL* cluster elements).
        
        **Individual models**: 
        - *Logistic Regression*: Linear, interpretable coefficients
        - *Random Forest*: Captures interactions, robust to outliers  
        - *XGBoost*: Gradient boosting, often highest individual performance
        
        See Supplementary Table S5 for performance metrics (ROC-AUC, Average Precision, 
        Precision@K) across 20 repeated train-test splits.
        """)
    
    prob_label = st.selectbox(
        "Probability column for ranking",
        prob_cols,
        index=0,
        help="Choose which model's predictions to use for ranking"
    )
    
    st.divider()
    st.subheader("📊 Filters")
    
    # Probability threshold
    pmin = st.slider(
        "Minimum pFOX threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Only include genes with predicted FOX probability above this threshold"
    )
    
    # Conservation filters (restored from v2)
    st.markdown("**Conservation Filters**")
    
    # Add RBH criteria explanation
    with st.expander("ℹ️ Filter criteria details (see Table S4)", expanded=False):
        st.markdown("""
        **Reciprocal Best Hit (RBH) Criteria:**
        - BLASTp identity ≥30%
        - Query coverage ≥70%
        - Subject coverage ≥70%
        - E-value ≤1e-10
        
        **Reference proteomes** (see Supplementary Table S3):
        - *Filamentous diazotrophs*: *Anabaena variabilis* ATCC 29413, 
          *Nostoc azollae* 0708, *Nostoc punctiforme* PCC 73102
        - *Unicellular diazotrophs*: *Cyanothece* sp. ATCC 51142, 
          *Crocosphaera watsonii* WH8501, *Gloeothece* sp. 6803
        - *Non-diazotrophs*: *Synechococcus elongatus* PCC 7942, 
          *Arthrospira maxima* CS-328
        """)
    
    fil_opts = st.multiselect(
        "Filamentous diazotroph hits",
        ["Conserved", "Not conserved"],
        default=["Conserved", "Not conserved"],
        help="Conserved = ≥1 RBH hit (≥30% identity) across filamentous diazotroph panel"
    )
    
    nd_opts = st.multiselect(
        "Non-diazotroph exclusion",
        ["Hit", "No hit"],
        default=["Hit", "No hit"],
        help="Hit = gene has ≥30% identity RBH to a non-diazotroph (may want to exclude)"
    )
    
    croco_opts = st.multiselect(
        "Cyanothece ATCC 51142 hit",
        ["Yes", "No"],
        default=["Yes", "No"],
        help="Yes = ≥30% identity RBH to the unicellular diazotroph ATCC 51142"
    )
    
    st.divider()
    st.subheader("🧬 Complement Design")
    
    # Length budget
    if "Gene length" in df.columns and df["Gene length"].notna().any():
        max_len = int(df["Gene length"].fillna(0).sum())
        nt_limit = st.number_input(
            "Length budget (nt, CDS only)",
            min_value=1000,
            max_value=max_len,
            value=min(50_000, max_len),
            step=1000,
            help="Maximum total coding sequence length. Does NOT include promoters, terminators, linkers, or vector backbone. Default 50 kb reflects stable payloads transferred to Synechocystis 6803."
        )
    else:
        nt_limit = 50_000

##############################################################################
# Apply Filters
##############################################################################

# Build filter mask
mask = (
    (df[prob_label].fillna(0) >= pmin) &
    (df["Filamentous_cons"].isin(fil_opts) if fil_opts else True) &
    (df["ND_cons"].isin(nd_opts) if nd_opts else True) &
    (df["Cyano_hit"].isin(croco_opts) if croco_opts else True)
)
flt = df[mask].copy()

# Summary statistics
st.subheader("📈 Dataset Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Genes after filtering", f"{len(flt):,}")
with col2:
    st.metric("Mean pFOX", f"{flt[prob_label].mean():.3f}" if len(flt) > 0 else "N/A")
with col3:
    high_conf = len(flt[flt[prob_label] >= 0.7])
    st.metric("High confidence (≥0.7)", f"{high_conf:,}")
with col4:
    total_nt = int(flt["Gene length"].sum()) if "Gene length" in flt.columns else 0
    st.metric("Total CDS", f"{total_nt:,} nt")

##############################################################################
# Gene Search (new feature)
##############################################################################

with st.expander("🔍 Gene Search", expanded=False):
    search_term = st.text_input(
        "Search by gene name, annotation, or product",
        placeholder="e.g., nifH, glycosyltransferase, all1234"
    )
    
    if search_term:
        search_cols = ["Annotation", "gene_symbol", "product", "Protein_names", "locus_tag"]
        search_mask = pd.Series([False] * len(flt), index=flt.index)
        for col in search_cols:
            if col in flt.columns:
                search_mask |= flt[col].astype(str).str.contains(search_term, case=False, na=False)
        
        search_results = flt[search_mask]
        
        if len(search_results) > 0:
            st.success(f"Found {len(search_results)} matching gene(s)")
            display_cols = [c for c in ["Annotation", "gene_symbol", "product", prob_label, 
                                        "Gene length", cyano_col] if c in search_results.columns]
            st.dataframe(
                search_results[display_cols].sort_values(prob_label, ascending=False),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No genes found matching your search")

##############################################################################
# Model Comparison (from manuscript ablation analysis)
##############################################################################

with st.expander("📊 Model Comparison: With vs Without Position Features", expanded=False):
    pos_col = "FOX probability"  # with position
    nopos_col = "prob_ensemble_mean_no_position"
    
    if pos_col in flt.columns and nopos_col in flt.columns:
        st.markdown("""
        The manuscript describes an ablation analysis comparing models trained with and without 
        genomic position features (`chromosome_region_start/end`). Some genes (like the *hupL* 
        cluster) show improved predictions when position encoding is removed because their 
        genomic context is disrupted during heterocyst differentiation.
        
        **Note on probability interpretation**: These scores are derived from classifier outputs 
        (originally in log-odds space for tree-based models) and are best interpreted as 
        *relative rankings* rather than calibrated probabilities. The difference (Δ pFOX) 
        indicates which genes are ranked differently by the two model variants.
        """)
        
        # Find genes with biggest differences
        flt_compare = flt[[c for c in ["Annotation", "gene_symbol", "product", pos_col, nopos_col, 
                                        "Gene length"] if c in flt.columns]].copy()
        flt_compare["Δ pFOX (no_pos - with_pos)"] = flt_compare[nopos_col] - flt_compare[pos_col]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Genes ranking higher WITHOUT position:**")
            top_nopos = flt_compare.nlargest(10, "Δ pFOX (no_pos - with_pos)")
            st.dataframe(top_nopos, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**Genes ranking higher WITH position:**")
            top_withpos = flt_compare.nsmallest(10, "Δ pFOX (no_pos - with_pos)")
            st.dataframe(top_withpos, hide_index=True, use_container_width=True)
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(flt_compare[pos_col], flt_compare[nopos_col], alpha=0.4, s=10)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel("pFOX with position features")
        ax.set_ylabel("pFOX without position features")
        ax.set_title("Model Comparison: Position Ablation")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Model comparison requires both 'FOX probability' and 'prob_ensemble_mean_no_position' columns.")

##############################################################################
# Build Complements
##############################################################################

st.divider()
st.subheader("🎯 Complement Design")

# Method explanation
with st.expander("ℹ️ About the selection methods", expanded=False):
    st.markdown("""
    **Rank Order Selection**: Adds genes in descending order of predicted FOX probability (pFOX) 
    until the nucleotide budget is exhausted. This method prioritizes the highest-confidence 
    candidates but may include larger genes that consume more of the budget.
    
    **Greedy Optimization**: Adds genes by descending pFOX-per-length ratio, maximizing the 
    total expected FOX probability within the size constraint. This approach tends to include 
    more smaller genes, achieving higher cumulative pFOX but potentially with more uncertainty 
    per individual gene.
    
    ---
    
    **⚠️ Important Interpretive Notes:**
    
    - **Probabilities are ranking heuristics**: The pFOX scores are classifier outputs (derived 
      from log-odds space) intended for *ranking* candidates, not as precisely calibrated 
      biological probabilities. Summing pFOX values provides a heuristic for expected FOX gene 
      count, not a guarantee.
    
    - **Length budget is CDS-only**: The nucleotide budget applies to coding sequence (CDS) 
      length only and does **not** include regulatory elements, promoters, terminators, linkers, 
      or vector backbone that would be required for actual construct assembly.
    
    - **Complements supplement the nif cluster**: These candidate sets are designed to 
      supplement the core *nif* regulon (assumed to be included separately), not replace it.
    
    **Reference**: See manuscript Supplementary Table S4 for complete filter definitions.
    """)

# Rank order complement
rank_order = cumulative_select(
    flt.sort_values(prob_label, ascending=False),
    sort_col=prob_label,
    length_col="Gene length",
    nt_lim=int(nt_limit),
    dedupe_col="Annotation"
)

# Greedy (prob per length) complement
ppl_col = "Prob_per_len" if "Prob_per_len" in flt.columns else "Prob_per_kb"
if ppl_col in flt.columns:
    greedy_opt = cumulative_select(
        flt.sort_values(ppl_col, ascending=False),
        sort_col=ppl_col,
        length_col="Gene length",
        nt_lim=int(nt_limit),
        dedupe_col="Annotation"
    )
else:
    greedy_opt = pd.DataFrame()

# Calculate expected FOX counts
exp_rank = round(rank_order[prob_label].sum(), 1) if len(rank_order) > 0 else 0
exp_greedy = round(greedy_opt[prob_label].sum(), 1) if len(greedy_opt) > 0 else 0

# Venn diagram
st.markdown("### Complement Overlap")

key_col = "Annotation" if "Annotation" in df.columns else None
if key_col and not rank_order.empty and not greedy_opt.empty:
    set_rank = set(rank_order[key_col].astype(str))
    set_greedy = set(greedy_opt[key_col].astype(str))
    
    col_venn, col_stats = st.columns([2, 1])
    
    with col_venn:
        fig, ax = plt.subplots(figsize=(6, 5))
        v = venn2([set_rank, set_greedy], ("Rank Order", "Greedy Optimization"), ax=ax)
        
        # Style the venn diagram
        for idx, color in enumerate(['#3498db', '#2ecc71']):
            if v.get_patch_by_id(['10', '01'][idx]):
                v.get_patch_by_id(['10', '01'][idx]).set_color(color)
                v.get_patch_by_id(['10', '01'][idx]).set_alpha(0.6)
        if v.get_patch_by_id('11'):
            v.get_patch_by_id('11').set_color('#9b59b6')
            v.get_patch_by_id('11').set_alpha(0.6)
        
        ax.set_title(
            f"Sum of pFOX\nRank: {exp_rank} | Greedy: {exp_greedy}",
            fontweight="bold",
            fontsize=12,
            pad=15
        )
        st.pyplot(fig)
        plt.close(fig)
    
    with col_stats:
        st.markdown("**Complement Statistics**")
        overlap = set_rank & set_greedy
        rank_only = set_rank - set_greedy
        greedy_only = set_greedy - set_rank
        union = set_rank | set_greedy
        
        st.write(f"• Rank order only: **{len(rank_only)}** genes")
        st.write(f"• Greedy only: **{len(greedy_only)}** genes")
        st.write(f"• Overlap: **{len(overlap)}** genes")
        st.write(f"• Union: **{len(union)}** genes")
        
        # Download options for set operations
        st.markdown("---")
        st.markdown("**Download gene sets:**")
        
        # Union genes
        union_df = flt[flt[key_col].astype(str).isin(union)].copy()
        union_df = union_df.sort_values(prob_label, ascending=False)
        download_csv(union_df, "union_both_methods")
        
        # Intersection genes
        if overlap:
            overlap_df = flt[flt[key_col].astype(str).isin(overlap)].copy()
            overlap_df = overlap_df.sort_values(prob_label, ascending=False)
            download_csv(overlap_df, "intersection_both_methods")
else:
    st.info("Venn diagram requires both complements to be non-empty with 'Annotation' column")

# Word clouds
st.markdown("### Functional Word Clouds")
overall = set(flt.get("Protein_names", pd.Series([], dtype=str)).dropna().apply(collapse_name).unique())

wc1, wc2 = st.columns(2)
with wc1:
    st.markdown("**Rank Order Selection**")
    if "Protein_names" in rank_order.columns and len(rank_order) > 0:
        make_wordcloud(rank_order["Protein_names"], "Rank Order", overall)
    else:
        st.info("No protein names available")

with wc2:
    st.markdown("**Greedy Optimization**")
    if not greedy_opt.empty and "Protein_names" in greedy_opt.columns:
        make_wordcloud(greedy_opt["Protein_names"], "Greedy", overall)
    else:
        st.info("No protein names available or Prob_per_len column missing")

##############################################################################
# Complement Tables
##############################################################################

st.markdown("### Complement Gene Tables")

left, right = st.columns(2)

with left:
    rank_len = int(rank_order["Gene length"].sum()) if "Gene length" in rank_order.columns else 0
    st.markdown(f"**Rank Order Selection** — {len(rank_order)} genes, {rank_len:,} nt")
    
    if len(rank_order) > 0:
        display_rank = enforce_col_order(rank_order)
        st.dataframe(display_rank, hide_index=True, use_container_width=True, height=400)
        download_csv(rank_order, "rank_order_complement")
    else:
        st.warning("No genes selected with current filters")

with right:
    if greedy_opt.empty:
        st.markdown("**Greedy Optimization** — *(Prob_per_len not available)*")
    else:
        greedy_len = int(greedy_opt["Gene length"].sum()) if "Gene length" in greedy_opt.columns else 0
        st.markdown(f"**Greedy Optimization** — {len(greedy_opt)} genes, {greedy_len:,} nt")
        
        display_greedy = enforce_col_order(greedy_opt)
        st.dataframe(display_greedy, hide_index=True, use_container_width=True, height=400)
        download_csv(greedy_opt, "greedy_optimization_complement")

##############################################################################
# Ortholog Panel Viewer (from v4)
##############################################################################

st.divider()
st.subheader("🔬 Ortholog Panel Viewer")

# Reference proteome panel info (addresses Reviewer 1 Major #1)
with st.expander("📋 Reference proteome panel (see Supplementary Table S3)", expanded=False):
    st.markdown("""
    **Comparative bioinformatics reference panel:**
    
    | Category | Organism | Strain/Accession |
    |----------|----------|------------------|
    | Filamentous diazotroph | *Anabaena variabilis* | ATCC 29413 |
    | Filamentous diazotroph | *Nostoc azollae* | 0708 |
    | Filamentous diazotroph | *Nostoc punctiforme* | PCC 73102 |
    | Unicellular diazotroph | *Cyanothece* sp. | ATCC 51142 |
    | Unicellular diazotroph | *Crocosphaera watsonii* | WH8501 |
    | Unicellular diazotroph | *Gloeothece* sp. | 6803 |
    | Non-diazotroph (unicellular) | *Synechococcus elongatus* | PCC 7942 |
    | Non-diazotroph (filamentous) | *Arthrospira maxima* | CS-328 |
    
    **RBH criteria**: BLASTp identity ≥30%, query/subject coverage ≥70%, E-value ≤1e-10
    
    *Full accession details and database sources are provided in Supplementary Table S3.*
    """)

with st.expander("View per-organism RBH hits for individual genes", expanded=False):
    # Check if ortholog columns exist
    has_ortholog_cols = any(c.endswith("__Prot") for c in flt.columns) or any(c.startswith("s_id__") for c in flt.columns)
    
    if has_ortholog_cols and key_col:
        gene_choices = sorted(flt[key_col].astype(str).unique())
        
        if gene_choices:
            selected_gene = st.selectbox(
                "Select gene (Annotation / locus tag)",
                gene_choices,
                index=0
            )
            
            gene_row = flt.loc[flt[key_col].astype(str) == str(selected_gene)].iloc[0]
            
            # Show gene info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Gene:** {selected_gene}")
                if "product" in gene_row.index:
                    st.write(f"**Product:** {gene_row['product']}")
            with col2:
                st.write(f"**pFOX:** {gene_row[prob_label]:.3f}")
                if "Gene length" in gene_row.index:
                    st.write(f"**Length:** {int(gene_row['Gene length'])} nt")
            with col3:
                if cyano_col in gene_row.index and pd.notna(gene_row[cyano_col]):
                    st.write(f"**Cyanothece %ID:** {gene_row[cyano_col]:.1f}%")
                if "filamentous_diazotroph_hits" in gene_row.index:
                    st.write(f"**Fil. diazo hits:** {int(gene_row['filamentous_diazotroph_hits'])}")
            
            # Show ortholog panel
            panel = extract_ortholog_panel(gene_row)
            if panel.empty:
                st.info("No RBH panel hits available for this gene")
            else:
                st.markdown("**Reciprocal Best Hits across reference panel:**")
                st.dataframe(panel, hide_index=True, use_container_width=True)
        else:
            st.info("No selectable gene identifiers found")
    else:
        st.info("Ortholog panel requires per-organism RBH columns (e.g., *__Prot). "
                "Use the updated app table with panel protein IDs.")

##############################################################################
# Footer
##############################################################################

st.divider()

# About section addressing key reviewer concerns
with st.expander("📖 About this tool & Supplementary Materials", expanded=False):
    st.markdown("""
    **FOX-Gene Complement Explorer** accompanies the manuscript:
    
    > Young J, Gu L, Zhou R. "Predicting FOX gene candidates for oxic nitrogen fixation 
    > using multi-omic machine learning and comparative bioinformatics."
    
    **Key Supplementary Tables:**
    - **S1**: Complete ranked gene list with all predictions and conservation metrics
    - **S2**: Feature importance matrix across all models
    - **S3**: Reference proteome panel with accession numbers and database sources
    - **S4**: Filter definitions and default values used in this app
    - **S5-S6**: Model performance metrics and cross-validation results
    - **S7**: Feature definitions and time-point mappings
    
    **Model Details:**
    - Trained on 68 literature-validated FOX genes (positive class) and 835 conserved 
      non-essential genes (negative proxy class) from *Anabaena* sp. PCC 7120
    - Features: RNA-seq (0/6/12/21 h post nitrogen step-down), proteomics, 
      promoter architecture, genomic context, and RBH conservation
    - Ensemble of Logistic Regression, Random Forest, and XGBoost classifiers
    - ROC-AUC ~0.78, Average Precision ~0.55 (see Table 1, Supplementary Table S5)
    
    **Important Caveats:**
    - Predictions are ranking heuristics, not calibrated probabilities
    - Experimental validation is required for any specific candidate
    - Length budgets are CDS-only (excluding regulatory elements)
    """)

st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    FOX-Gene Complement Explorer v5.0 | Predicting genes for oxic nitrogen fixation in <i>Anabaena</i> sp. PCC 7120<br>
    Young, Gu & Zhou — South Dakota State University<br>
    <a href="https://github.com/jamesyoung93/FoxGenesApp" target="_blank">Source Code</a> | 
    <a href="https://github.com/jamesyoung93/FoxGenes_ML" target="_blank">ML Pipeline</a>
    </div>
    """,
    unsafe_allow_html=True
)
