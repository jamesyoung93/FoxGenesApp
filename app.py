# Streamlit app for exploring predicted FOX genes in *Anabaena* 7120
# v5 – Crocosphaera join  +  ENS_PRED → FOX probability
import io, re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib_venn import venn2
from wordcloud import WordCloud, STOPWORDS

###############################################################################
# ---------------------------- CONFIGURATION ----------------------------------
###############################################################################
DATA_PATH  = Path(__file__).with_name("FOX_unknown_with_hits_function_greedy.csv")
CROC_PATH  = Path(__file__).with_name("filamentous_specific.csv")       # NEW
CROC_COLS  = {                                                          # NEW
    "croco_protein_name": "Croco_Prot",
    "croco_pct_identity": "Croco_%ID"
}
WC_SEED = 42            # reproducible word-clouds

###############################################################################
# ----------------------------- HELPERS  --------------------------------------
###############################################################################
def load_data(path: Path | str) -> pd.DataFrame:
    @st.cache_data(show_spinner=False)
    def _read(p: str) -> pd.DataFrame:
        return pd.read_csv(p)
    return _read(str(path))

# ------------- word-cloud helpers (unchanged – snipped for brevity) ----------
EXTRA_STOP = {...}
STOPWORDS_FULL = STOPWORDS.union(EXTRA_STOP)
def collapse_name(name: str) -> str: ...
def make_wordcloud(series: pd.Series, title: str, overall_collapsed_set: set): ...

# -------------- complement helpers (cumulative_select, etc.) -----------------
def cumulative_select(df: pd.DataFrame, sort_col: str, limit_nt: int) -> pd.DataFrame: ...
def enforce_col_order(tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure column order:
      Prob_per_len | Protein_names | Croco_Prot | Croco_%ID | ...
    """
    cols = list(tbl.columns)
    wanted = ["Protein_names"] + list(CROC_COLS.values())
    if "Protein_names" in cols:
        base_idx = cols.index("Protein_names")
        # pull any Crocosphaera columns out & re-insert right after Protein_names
        for w in wanted[1:][::-1]:
            if w in cols:
                cols.insert(base_idx + 1, cols.pop(cols.index(w)))
    if "Prob_per_len" in cols:
        prob_idx = cols.index("Prob_per_len")
        cols.insert(prob_idx + 1, cols.pop(cols.index("Protein_names")))
    return tbl[cols]

def download_csv(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(),
                       file_name=f"{label}.csv", mime="text/csv")

###############################################################################
# ----------------------------- UI / LOGIC ------------------------------------
###############################################################################
st.set_page_config(page_title="FOX-Gene Complement Explorer", layout="wide")
st.title("FOX-Gene Complement Explorer")

# ---------- 1. Load primary data --------------------------------------------
if DATA_PATH.exists():
    df = load_data(DATA_PATH)
else:
    st.warning("Upload your merged CSV to begin")
    up = st.file_uploader("Merged CSV file", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)

# ---------- 2. OPTIONAL: join Crocosphaera hits -----------------------------
if CROC_PATH.exists():
    croc = load_data(CROC_PATH).rename(columns=CROC_COLS)
    df = df.merge(croc, on="Annotation", how="left")

# ---------- 3. Basic cleanup & typing ---------------------------------------
numeric_cols = ["ENS_PRED", "Gene length", "Prob_per_len",
                "non_diazotroph_hits", "filamentous_diazotroph_hits"]
for col in numeric_cols:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create conservation flags (unchanged)
...

# ----------------------------- Sidebar --------------------------------------
...

# ----------------- Build complements (rank & greedy) ------------------------
rank_order = cumulative_select(flt.sort_values("ENS_PRED", ascending=False),
                               "ENS_PRED", nt_limit)
greedy_opt = cumulative_select(flt.sort_values("Prob_per_len", ascending=False),
                               "Prob_per_len", nt_limit)

# --- OPTIONAL duplicate-collapse to align row & Venn counts (unchanged) -----
rank_order  = rank_order.drop_duplicates("Annotation")
greedy_opt  = greedy_opt.drop_duplicates("Annotation")

# ---------- 4. Add Crocosphaera info ONLY to greedy complement -------------
if CROC_PATH.exists():
    greedy_opt = greedy_opt.merge(croc, on="Annotation", how="left")

# ---------- 5. Rename ENS_PRED for user-facing outputs ----------------------
rank_order  = rank_order.rename(columns={"ENS_PRED": "FOX probability"})
greedy_opt  = greedy_opt.rename(columns={"ENS_PRED": "FOX probability"})
flt         = flt.rename(columns={"ENS_PRED": "FOX probability"})

# ---------- 6. Re-order columns --------------------------------------------
rank_order  = enforce_col_order(rank_order)
greedy_opt  = enforce_col_order(greedy_opt)

# ---------- 7. Venn diagram & expected FOX counts ---------------------------
set_rank   = set(rank_order["Annotation"])
set_greedy = set(greedy_opt["Annotation"])
exp_rank   = round(rank_order["FOX probability"].sum())
exp_greedy = round(greedy_opt["FOX probability"].sum())

st.markdown("### Overlap between complements (with expected FOX genes)")
venn_col = st.columns([1, 2, 1])[1]
with venn_col:
    fig, ax = plt.subplots(figsize=(4, 4))
    venn2([set_rank, set_greedy],
          ("Rank Order Selection", "Greedy Optimization"), ax=ax)
    ax.set_title(f"Expected FOX genes → Rank: {exp_rank} | Greedy: {exp_greedy}",
                 fontweight="bold", pad=20)
    st.pyplot(fig)

# ---------- 8. Word clouds ---------------------------------------------------
...

# ---------- 9. Display tables & allow download ------------------------------
st.markdown("### Complement tables")
left, right = st.columns(2)
with left:
    st.markdown(f"#### Rank Order Selection — {len(rank_order)} genes, "
                f"{int(rank_order['Gene length'].sum()):,} nt")
    st.dataframe(rank_order, hide_index=True, use_container_width=True)
    download_csv(rank_order, "rank_order_selection")

with right:
    st.markdown(f"#### Greedy Optimization — {len(greedy_opt)} genes, "
                f"{int(greedy_opt['Gene length'].sum()):,} nt")
    st.dataframe(greedy_opt, hide_index=True, use_container_width=True)
    download_csv(greedy_opt, "greedy_optimization")

st.markdown("---")
st.write("Filtered universe:", len(flt), "genes of", len(df))
