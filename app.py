# FOX-Gene Complement Explorer  –  single-CSV version
#
# Requires:  FOX_unknown_with_hits_function_greedy_enriched.csv
# ---------------------------------------------------------------------------

import io, re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib_venn import venn2
from wordcloud import WordCloud, STOPWORDS

##############################################################################
# ------------------------------ HELPERS ------------------------------------#
##############################################################################

def load_data(path: Path | str) -> pd.DataFrame:
    @st.cache_data(show_spinner=False)
    def _read(p): return pd.read_csv(p)
    return _read(str(path))


# ---------- word-cloud helpers ---------------------------------------------
EXTRA_STOP = {
    "protein", "putative", "family", "domain", "predicted", "hypothetical",
    "probable", "possible", "like", "related"
}
STOPWORDS_FULL = STOPWORDS.union(EXTRA_STOP)
WC_SEED = 42


def collapse_name(name: str) -> str:
    """Collapse locus-tag-style names to 'Unknown' and all ribosomal variants to 'Ribosomal'."""
    if pd.isna(name):
        return "Unknown"
    low = name.lower()
    if re.match(r"^(all|alr|asl|asr)\d+", name, re.I):
        return "Unknown"
    if "ribosom" in low:
        return "Ribosomal"
    return name


def make_wordcloud(series: pd.Series, title: str, overall_collapsed: set):
    collapsed = series.dropna().apply(collapse_name)
    unique = set(collapsed.unique())
    if "Unknown" in overall_collapsed and "Unknown" not in unique:
        unique.add("Unknown")
    if not unique:
        st.write(f"*(no names in {title})*")
        return

    txt = " ".join(unique)
    wc = WordCloud(
        width=800, height=300, background_color="white",
        stopwords=STOPWORDS_FULL, random_state=WC_SEED
    ).generate(txt)

    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# ---------- complement utilities -------------------------------------------
def cumulative_select(df: pd.DataFrame, sort_col: str, limit_nt: int) -> pd.DataFrame:
    sel, cum = [], 0
    for _, row in df.iterrows():
        glen = row["Gene length"]
        if pd.isna(glen):
            continue
        if cum + glen > limit_nt:
            break
        cum += glen
        sel.append(row)
    return pd.DataFrame(sel)


def enforce_col_order(tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Order: Prob_per_len | Protein_names | Croco_Prot | Croco_%ID | …rest…
    """
    cols = list(tbl.columns)

    # Move Croco columns immediately after Protein_names
    for c in ["Croco_%ID", "Croco_Prot"][::-1]:
        if c in cols and "Protein_names" in cols:
            cols.insert(cols.index("Protein_names") + 1, cols.pop(cols.index(c)))

    # Ensure Protein_names follows Prob_per_len
    if "Protein_names" in cols and "Prob_per_len" in cols:
        cols.insert(cols.index("Prob_per_len") + 1, cols.pop(cols.index("Protein_names")))

    return tbl[cols]


def download_csv(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(f"Download {label}", buf.getvalue(),
                       file_name=f"{label}.csv", mime="text/csv")


##############################################################################
# ------------------------------- APP ---------------------------------------#
##############################################################################

st.set_page_config(page_title="FOX-Gene Complement Explorer", layout="wide")
st.title("FOX-Gene Complement Explorer")

# ---------- Load CSV --------------------------------------------------------
DATA_PATH = Path(__file__).with_name("FOX_unknown_with_hits_function_greedy_enriched.csv")
if DATA_PATH.exists():
    df = load_data(DATA_PATH)
else:
    st.warning("Upload your enriched CSV to begin")
    up = st.file_uploader("Enriched CSV", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)

# ---------- Rename ENS_PRED -> FOX probability straight away ---------------
if "ENS_PRED" in df.columns:
    df.rename(columns={"ENS_PRED": "FOX probability"}, inplace=True)

# ---------- Ensure numeric columns -----------------------------------------
for col in ["FOX probability", "Gene length", "Prob_per_len",
            "non_diazotroph_hits", "filamentous_diazotroph_hits"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- Conservation flags ---------------------------------------------
df["Filamentous_cons"] = df["filamentous_diazotroph_hits"]\
                            .notna().map({True: "Conserved", False: "Not conserved"})
nd_hit = df["non_diazotroph_hits"].notna() & (df["non_diazotroph_hits"] > 0)
df["ND_cons"] = nd_hit.map({True: "Hit", False: "No hit"})

# ---------- Sidebar filters -------------------------------------------------
with st.expander("🔍 Filter Options", expanded=True):
    st.header("Filters")
    fil_opts = st.multiselect("Filamentous conservation",
                              ["Conserved", "Not conserved"],
                              default=["Conserved", "Not conserved"])
    nd_opts = st.multiselect("Non-diazotroph hit", ["Hit", "No hit"],
                             default=["Hit", "No hit"])
    nt_limit = st.number_input("Complement length limit (nt)",
                               min_value=1000,
                               max_value=int(df["Gene length"].sum()),
                               value=50_000, step=1000)

flt = df[df["Filamentous_cons"].isin(fil_opts) &
         df["ND_cons"].isin(nd_opts)].copy()

# ---------- Build complements ----------------------------------------------
rank_order = cumulative_select(flt.sort_values("FOX probability", ascending=False),
                               "FOX probability", nt_limit).drop_duplicates("Annotation")

greedy_opt = cumulative_select(flt.sort_values("Prob_per_len", ascending=False),
                               "Prob_per_len", nt_limit).drop_duplicates("Annotation")

# ---------- Expected FOX counts --------------------------------------------
set_rank, set_greedy = set(rank_order["Annotation"]), set(greedy_opt["Annotation"])
exp_rank   = round(rank_order["FOX probability"].sum())
exp_greedy = round(greedy_opt["FOX probability"].sum())

# ---------- Word-cloud preparation -----------------------------------------
overall_collapsed = set(flt["Protein_names"].dropna().apply(collapse_name).unique())

# ---------- Layout ----------------------------------------------------------
st.markdown("### Overlap between complements (with expected FOX genes)")
venn_col = st.columns([1, 2, 1])[1]
with venn_col:
    fig, ax = plt.subplots(figsize=(4, 4))
    venn2([set_rank, set_greedy],
          ("Rank Order Selection", "Greedy Optimization"), ax=ax)
    ax.set_title(f"Expected FOX genes → Rank: {exp_rank} | Greedy: {exp_greedy}",
                 fontweight="bold", pad=20)
    st.pyplot(fig)

st.markdown("### Word-cloud comparison (unique collapsed names)")
wc1, wc2 = st.columns(2)
with wc1:
    st.caption("Rank Order Selection complement")
    make_wordcloud(rank_order["Protein_names"], "Rank Order", overall_collapsed)
with wc2:
    st.caption("Greedy Optimization complement")
    make_wordcloud(greedy_opt["Protein_names"], "Greedy Optimization", overall_collapsed)

# ---------- Enforce column order & display ----------------------------------
rank_order = enforce_col_order(rank_order)
greedy_opt = enforce_col_order(greedy_opt)

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
