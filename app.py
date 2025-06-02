# Streamlit app for exploring predicted FOX genes in *Anabaena* 7120

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib_venn import venn2
from wordcloud import WordCloud, STOPWORDS

################################################################################
# Helper utilities
################################################################################

def load_data(path: Path | str) -> pd.DataFrame:
    @st.cache_data(show_spinner=False)
    def _read(p: str) -> pd.DataFrame:
        return pd.read_csv(p)
    return _read(str(path))

# ------------------------ Word-cloud helpers ---------------------------------

EXTRA_STOP = {
    "protein", "putative", "family", "domain", "predicted", "hypothetical",
    "probable", "possible", "like", "related"
}
STOPWORDS_FULL = STOPWORDS.union(EXTRA_STOP)
WC_SEED = 42  # reproducible

def collapse_name(name: str) -> str:
    """
    Collapse any missing / identifier-like names into "Unknown".
    Everything else stays as-is.
    """
    if pd.isna(name):
        return "Unknown"
    # If it looks like a locus tag, collapse to "Unknown"
    if re.match(r"^(all|alr|asl|asr)\d+", name, re.IGNORECASE):
        return "Unknown"
    return name

def make_wordcloud(series: pd.Series, title: str):
    """
    Build a wordcloud from UNIQUE collapsed names (no duplicates).
    """
    # 1) Collapse each name to either its real value or "Unknown"
    collapsed = series.dropna().apply(collapse_name)
    # 2) Keep only unique values (so "Unknown" only appears once)
    unique_names = pd.Series(collapsed.unique())
    # 3) Join into one big string
    txt = " ".join(unique_names)
    if not txt:
        st.write(f"*(no names in {title})*")
        return

    wc = WordCloud(
        width=800,
        height=300,
        background_color="white",
        stopwords=STOPWORDS_FULL,
        random_state=WC_SEED,
    ).generate(txt)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# --------------------- Complement selection helpers -------------------------

def cumulative_select(df: pd.DataFrame, sort_col: str, limit_nt: int) -> pd.DataFrame:
    """
    Return rows in sort order until cumulative 'Gene length' <= limit_nt.
    """
    sel, cum_len = [], 0
    for _, row in df.iterrows():
        ln = row["Gene length"]
        if pd.isna(ln):
            continue
        if cum_len + ln > limit_nt:
            break
        cum_len += ln
        sel.append(row)
    return pd.DataFrame(sel)

def enforce_col_order(tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Move 'Protein_names' to immediately follow 'Prob_per_len'.
    """
    cols = list(tbl.columns)
    if "Protein_names" in cols and "Prob_per_len" in cols:
        # remove and re-insert
        idx_p = cols.index("Protein_names")
        idx_prob = cols.index("Prob_per_len")
        # pop out
        prot = cols.pop(idx_p)
        # insert right after Prob_per_len
        cols.insert(idx_prob + 1, prot)
        tbl = tbl[cols]
    return tbl

def download_csv(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=f"{label}.csv", mime="text/csv")

################################################################################
# Streamlit UI
################################################################################

st.set_page_config(page_title="FOX-Gene Complement Explorer", layout="wide")
st.title("FOX-Gene Complement Explorer")

DATA_PATH = Path(__file__).with_name("FOX_unknown_with_hits_function_greedy.csv")
if DATA_PATH.exists():
    df = load_data(DATA_PATH)
else:
    st.warning("Upload your merged CSV to begin")
    up = st.file_uploader("Merged CSV file")
    if up is None:
        st.stop()
    df = pd.read_csv(up)

# Ensure numeric columns are numeric
for col in ["ENS_PRED", "Gene length", "Prob_per_len", "non_diazotroph_hits", "filamentous_diazotroph_hits"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Conservation flags
df["Filamentous_cons"] = df["filamentous_diazotroph_hits"].notna().map({True: "Conserved", False: "Not conserved"})
nd_hit = df["non_diazotroph_hits"].notna() & (df["non_diazotroph_hits"] > 0)
df["ND_cons"] = nd_hit.map({True: "Hit", False: "No hit"})

# ----------------------------- Sidebar --------------------------------------
with st.expander("🔍 Filter Options", expanded=True):
    st.header("Filters")
    fil_opts = st.multiselect("Filamentous conservation", ["Conserved", "Not conserved"], ["Conserved", "Not conserved"])
    nd_opts = st.multiselect("Non-diazotroph hit", ["Hit", "No hit"], ["Hit", "No hit"])
    nt_limit = st.number_input("Complement length limit (nt)",
                               min_value=1000,
                               max_value=int(df["Gene length"].sum()),
                               value=50000,
                               step=1000)

# Apply filters
mask = df["Filamentous_cons"].isin(fil_opts) & df["ND_cons"].isin(nd_opts)
flt = df[mask].copy()

# Build complements
rank_order = cumulative_select(flt.sort_values("ENS_PRED", ascending=False), "ENS_PRED", nt_limit)
greedy_opt = cumulative_select(flt.sort_values("Prob_per_len", ascending=False), "Prob_per_len", nt_limit)

# Build sets for Venn
set_rank = set(rank_order["Annotation"])
set_greedy = set(greedy_opt["Annotation"])

# Compute rounded expected FOX genes
exp_rank = round(rank_order["ENS_PRED"].sum())
exp_greedy = round(greedy_opt["ENS_PRED"].sum())

################################################################################
# ----------------------------- Layout ---------------------------------------
################################################################################

# Venn diagram + expected FOX genes in one figure
st.markdown("### Overlap between complements (with expected FOX genes)")
venn_col = st.columns([1, 2, 1])[1]
with venn_col:
    fig, ax = plt.subplots(figsize=(4, 4))

    # Draw Venn
    venn = venn2([set_rank, set_greedy], ("Rank Order Selection", "Greedy Optimization"), ax=ax)

    # Title that includes expected FOX genes for each approach
    ax.set_title(f"Expected FOX genes → Rank: {exp_rank} | Greedy: {exp_greedy}", fontweight="bold")

    st.pyplot(fig)

# Word clouds (side by side)
st.markdown("### Word-cloud comparison (unique names only)")
wc1, wc2 = st.columns(2)
with wc1:
    st.caption("Rank Order Selection complement")
    make_wordcloud(rank_order["Protein_names"], "Rank Order")
with wc2:
    st.caption("Greedy Optimization complement")
    make_wordcloud(greedy_opt["Protein_names"], "Greedy")

# Reorder columns so that 'Protein_names' follows 'Prob_per_len'
rank_order = enforce_col_order(rank_order)
greedy_opt = enforce_col_order(greedy_opt)

# Complement tables
st.markdown("### Complement tables")
left, right = st.columns(2)
with left:
    st.markdown(f"#### Rank Order Selection — {len(rank_order)} genes, {int(rank_order['Gene length'].sum()):,} nt")
    st.dataframe(rank_order, hide_index=True, use_container_width=True)
    download_csv(rank_order, "rank_order_selection")

with right:
    st.markdown(f"#### Greedy Optimization — {len(greedy_opt)} genes, {int(greedy_opt['Gene length'].sum()):,} nt")
    st.dataframe(greedy_opt, hide_index=True, use_container_width=True)
    download_csv(greedy_opt, "greedy_optimization")

st.markdown("---")
st.write("Filtered universe:", len(flt), "genes of", len(df))
