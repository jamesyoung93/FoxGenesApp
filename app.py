# FOX-Gene Complement Explorer (v5)
#
# • Joins Crocosphaera sp. RS NRE 51142 protein name + %ID onto GREEDY complement
# • Renames ENS_PRED → “FOX probability” everywhere user facing
# • Keeps Crocosphaera columns just right of Protein_names
#
# ---------------------------------------------------------------------------

import io, re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib_venn import venn2
from wordcloud import WordCloud, STOPWORDS


##############################################################################
# ----------------------------- CONFIGURATION -------------------------------#
##############################################################################

PRIMARY_CSV      = "FOX_unknown_with_hits_function_greedy.csv"
CROCO_CSV        = "filamentous_specific.csv"         # optional
CROCO_COL_RENAME = {                                  # incoming → pretty
    "croco_protein_name": "Croco_Prot",
    "croco_pct_identity": "Croco_%ID"
}

WC_SEED = 42  # reproducible word-clouds


##############################################################################
# ------------------------------ HELPERS ------------------------------------#
##############################################################################

def load_csv(path: Path | str) -> pd.DataFrame:
    @st.cache_data(show_spinner=False)
    def _read(p: str):
        return pd.read_csv(p)
    return _read(str(path))


# ----- Word-cloud helpers ---------------------------------------------------

EXTRA_STOP = {
    "protein", "putative", "family", "domain", "predicted", "hypothetical",
    "probable", "possible", "like", "related"
}
STOPWORDS_FULL = STOPWORDS.union(EXTRA_STOP)


def collapse_name(name: str) -> str:
    """Collapse locus-tag-style names to ‘Unknown’ and all ribosomal variants to ‘Ribosomal’."""
    if pd.isna(name):
        return "Unknown"

    low = name.lower()
    if re.match(r"^(all|alr|asl|asr)\d+", name, re.IGNORECASE):
        return "Unknown"
    if "ribosom" in low:
        return "Ribosomal"
    return name


def make_wordcloud(series: pd.Series, title: str, all_collapsed: set):
    collapsed = series.dropna().apply(collapse_name)
    unique = set(collapsed.unique())

    # Guarantee exactly one “Unknown” if it appeared anywhere
    if "Unknown" in all_collapsed and "Unknown" not in unique:
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


# ----- Complement helpers ---------------------------------------------------

def cumulative_select(df: pd.DataFrame, sort_col: str, nt_limit: int) -> pd.DataFrame:
    """Take rows in descending order of sort_col until cumulative Gene length > nt_limit."""
    sel = []
    cum = 0
    for _, row in df.iterrows():
        glen = row["Gene length"]
        if pd.isna(glen):
            continue
        if cum + glen > nt_limit:
            break
        cum += glen
        sel.append(row)
    return pd.DataFrame(sel)


def enforce_col_order(tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Put Protein_names → Croco columns → (rest) and keep Protein_names right after Prob_per_len.
    """
    cols = list(tbl.columns)

    # Ensure Croco columns exist in the list even if NaN only
    for nice in CROCO_COL_RENAME.values():
        if nice in cols and "Protein_names" in cols:
            cols.insert(cols.index("Protein_names") + 1, cols.pop(cols.index(nice)))

    # Make Protein_names follow Prob_per_len
    if "Protein_names" in cols and "Prob_per_len" in cols:
        cols.insert(cols.index("Prob_per_len") + 1, cols.pop(cols.index("Protein_names")))

    return tbl[cols]


def download_csv(df: pd.DataFrame, label: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(f"Download {label}", buf.getvalue(),
                       file_name=f"{label}.csv", mime="text/csv")


##############################################################################
# -------------------------- STREAMLIT APP ----------------------------------#
##############################################################################

st.set_page_config(page_title="FOX-Gene Complement Explorer", layout="wide")
st.title("FOX-Gene Complement Explorer")

# ---------- Load primary data ----------------------------------------------
p_path = Path(__file__).with_name(PRIMARY_CSV)
if p_path.exists():
    df = load_csv(p_path)
else:
    st.warning("Upload your merged CSV to begin")
    up = st.file_uploader("Merged CSV", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)

# ---------- Optional Crocosphaera join -------------------------------------
c_path = Path(__file__).with_name(CROCO_CSV)
if c_path.exists():
    croco = load_csv(c_path).rename(columns=CROCO_COL_RENAME)
    df = df.merge(croco, on="Annotation", how="left")
else:
    croco = None

# ---------- Type safety ----------------------------------------------------
for col in ["ENS_PRED", "Gene length", "Prob_per_len",
            "non_diazotroph_hits", "filamentous_diazotroph_hits"]:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------- Conservation flags ---------------------------------------------
df["Filamentous_cons"] = df["filamentous_diazotroph_hits"].notna().map({True: "Conserved", False: "Not conserved"})
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

flt = df[df["Filamentous_cons"].isin(fil_opts) & df["ND_cons"].isin(nd_opts)].copy()

# ---------- Build complements ----------------------------------------------
rank_order = cumulative_select(flt.sort_values("ENS_PRED", ascending=False),
                               "ENS_PRED", nt_limit)
greedy_opt = cumulative_select(flt.sort_values("Prob_per_len", ascending=False),
                               "Prob_per_len", nt_limit)

rank_order = rank_order.drop_duplicates("Annotation")
greedy_opt = greedy_opt.drop_duplicates("Annotation")  # aligns Venn & table counts

# ---------- Attach Croco columns to GREEDY only ----------------------------
# (already merged in df; here just ensure presence even if empty)
if croco is not None:
    greedy_opt = greedy_opt  # nothing extra to do

# ---------- Rename ENS_PRED → FOX probability ------------------------------
for tbl in (rank_order, greedy_opt, flt):
    tbl.rename(columns={"ENS_PRED": "FOX probability"}, inplace=True)

# ---------- Re-order columns ----------------------------------------------
rank_order = enforce_col_order(rank_order)
greedy_opt = enforce_col_order(greedy_opt)

# ---------- Venn + expected counts -----------------------------------------
set_rank, set_greedy = set(rank_order["Annotation"]), set(greedy_opt["Annotation"])
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

# ---------- Word-clouds -----------------------------------------------------
overall_collapsed = set(flt["Protein_names"].dropna().apply(collapse_name).unique())

st.markdown("### Word-cloud comparison (unique collapsed names)")
wc1, wc2 = st.columns(2)
with wc1:
    st.caption("Rank Order Selection complement")
    make_wordcloud(rank_order["Protein_names"], "Rank Order", overall_collapsed)

with wc2:
    st.caption("Greedy Optimization complement")
    make_wordcloud(greedy_opt["Protein_names"], "Greedy Optimization", overall_collapsed)

# ---------- Tables + downloads ---------------------------------------------
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
