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
    Collapse any missing or locus‐tag–style names into "Unknown".
    Otherwise return the name unchanged.
    """
    if pd.isna(name):
        return "Unknown"
    # If it looks like a locus tag (e.g. “all0001”, “alr0345”, etc.), collapse to “Unknown”
    if re.match(r"^(all|alr|asl|asr)\d+", name, re.IGNORECASE):
        return "Unknown"
    return name

def make_wordcloud(
    series: pd.Series,
    title: str,
    overall_collapsed_set: set
):
    """
    Build a word cloud from unique collapsed names in `series`.  
    If "Unknown" is in overall_collapsed_set (i.e. present anywhere in the filtered universe),
    then force exactly one "Unknown" into this cloud—even if `series` had no "Unknown".
    """
    # 1) Collapse each name (or NaN) → “Unknown” if needed
    collapsed = series.dropna().apply(collapse_name)

    # 2) Take the unique collapsed names from this complement
    unique_names = set(collapsed.unique())

    # 3) If “Unknown” was collapsed anywhere in the full filtered universe
    #    (overall_collapsed_set), but not in this complement, add it here:
    if ("Unknown" in overall_collapsed_set) and ("Unknown" not in unique_names):
        unique_names.add("Unknown")

    # 4) Prepare the input text for WordCloud
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
    Return rows (in the order of descending `sort_col`) until
    the cumulative 'Gene length' would exceed `limit_nt`.
    """
    selected_rows = []
    cum_len = 0
    for _, row in df.iterrows():
        length = row["Gene length"]
        if pd.isna(length):
            continue
        if cum_len + length > limit_nt:
            break
        cum_len += length
        selected_rows.append(row)
    return pd.DataFrame(selected_rows)

def enforce_col_order(tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Move 'Protein_names' column to immediately follow 'Prob_per_len' if present.
    """
    cols = list(tbl.columns)
    if "Protein_names" in cols and "Prob_per_len" in cols:
        prot_idx = cols.index("Protein_names")
        prob_idx = cols.index("Prob_per_len")
        # pop out and re-insert right after Prob_per_len
        cols.insert(prob_idx + 1, cols.pop(prot_idx))
        tbl = tbl[cols]
    return tbl

def download_csv(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(label=label, data=buf.getvalue(), file_name=f"{label}.csv", mime="text/csv")


################################################################################
# Streamlit UI
################################################################################

st.set_page_config(page_title="FOX-Gene Complement Explorer", layout="wide")
st.title("FOX-Gene Complement Explorer")

# Attempt to load a CSV from the same directory; if not found, ask user to upload
DATA_PATH = Path(__file__).with_name("FOX_unknown_with_hits_function_greedy.csv")
if DATA_PATH.exists():
    df = load_data(DATA_PATH)
else:
    st.warning("Upload your merged CSV to begin")
    up = st.file_uploader("Merged CSV file", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)

# Ensure numeric columns are numeric
for col in ["ENS_PRED", "Gene length", "Prob_per_len", "non_diazotroph_hits", "filamentous_diazotroph_hits"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create conservation flags
df["Filamentous_cons"] = df["filamentous_diazotroph_hits"].notna().map({True: "Conserved", False: "Not conserved"})
nd_hit = df["non_diazotroph_hits"].notna() & (df["non_diazotroph_hits"] > 0)
df["ND_cons"] = nd_hit.map({True: "Hit", False: "No hit"})

# ----------------------------- Sidebar --------------------------------------

with st.expander("🔍 Filter Options", expanded=True):
    st.header("Filters")
    fil_opts = st.multiselect(
        "Filamentous conservation",
        options=["Conserved", "Not conserved"],
        default=["Conserved", "Not conserved"]
    )
    nd_opts = st.multiselect(
        "Non-diazotroph hit",
        options=["Hit", "No hit"],
        default=["Hit", "No hit"]
    )
    nt_limit = st.number_input(
        "Complement length limit (nt)",
        min_value=1000,
        max_value=int(df["Gene length"].sum()),
        value=50000,
        step=1000
    )

# Apply filters
mask = df["Filamentous_cons"].isin(fil_opts) & df["ND_cons"].isin(nd_opts)
flt = df[mask].copy()

# Build each complement:
#  - rank_order: sort by ENS_PRED descending
#  - greedy_opt: sort by Prob_per_len descending
rank_order = cumulative_select(flt.sort_values("ENS_PRED", ascending=False), "ENS_PRED", nt_limit)
greedy_opt   = cumulative_select(flt.sort_values("Prob_per_len", ascending=False), "Prob_per_len", nt_limit)

# ---- OPTION A: DROP DUPLICATE ANNOTATIONS TO ALIGN VENN COUNTS WITH ROW COUNTS ----
rank_order = rank_order.drop_duplicates(subset="Annotation")
greedy_opt = greedy_opt.drop_duplicates(subset="Annotation")

# Create sets of gene annotations for the Venn diagram
set_rank   = set(rank_order["Annotation"])
set_greedy = set(greedy_opt["Annotation"])

# Compute rounded expected FOX genes for each complement
exp_rank   = round(rank_order["ENS_PRED"].sum())
exp_greedy = round(greedy_opt["ENS_PRED"].sum())

# To know if “Unknown” should appear at least once:
#   collapse ALL Protein_names in flt → see if “Unknown” is in that set
all_collapsed       = flt["Protein_names"].dropna().apply(collapse_name)
overall_collapsed_set = set(all_collapsed.unique())

################################################################################
# ----------------------------- Layout ---------------------------------------
################################################################################

# 1) Venn diagram + expected FOX‐gene counts in one figure
st.markdown("### Overlap between complements (with expected FOX genes)")
venn_col = st.columns([1, 2, 1])[1]
with venn_col:
    fig, ax = plt.subplots(figsize=(4, 4))

    venn2(
        [set_rank, set_greedy],
        ("Rank Order Selection", "Greedy Optimization"),
        ax=ax
    )

    # Title includes both expected FOX counts
    ax.set_title(
        f"Expected FOX genes → Rank: {exp_rank} | Greedy: {exp_greedy}",
        fontweight="bold",
        pad=20
    )

    st.pyplot(fig)

# 2) Word clouds side-by-side. Each will force exactly one "Unknown" if it was
#    present anywhere in flt, but never more than once.
st.markdown("### Word-cloud comparison (unique collapsed names, pulling in 'Unknown' once)")
wc1, wc2 = st.columns(2)
with wc1:
    st.caption("Rank Order Selection complement")
    make_wordcloud(rank_order["Protein_names"], "Rank Order", overall_collapsed_set)
with wc2:
    st.caption("Greedy Optimization complement")
    make_wordcloud(greedy_opt["Protein_names"], "Greedy Optimization", overall_collapsed_set)

# 3) Reorder columns so that Protein_names follows Prob_per_len
rank_order = enforce_col_order(rank_order)
greedy_opt = enforce_col_order(greedy_opt)

# 4) Show the detailed complement tables
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
