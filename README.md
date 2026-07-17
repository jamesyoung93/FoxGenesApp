# FOX Gene Complement Explorer

[Open the live Streamlit app](https://foxgenesapp-h6mlszof5jdd6p2bpq22wr.streamlit.app/)

FOX Gene Complement Explorer turns the published pFOX candidate rankings into a reviewable design tool. Users can search and filter genes, compare conservation evidence across cyanobacteria, and assemble candidate accessory-gene complements under a coding-sequence length budget. It is intended to support experimental planning, not to replace genetic validation.

**Paper:** [Predicting FOX gene candidates for oxic nitrogen fixation using multi-omic machine learning and comparative bioinformatics](https://doi.org/10.1038/s41598-026-41873-w), *Scientific Reports* 16, 11412 (2026)

**Modeling code:** [FoxGenes_ML](https://github.com/jamesyoung93/FoxGenes_ML)

## Key result

The associated paper reports that the combined multi-omic models ranked held-out FOX genes above the 0.075 prevalence baseline and released genome-wide pFOX scores for candidate prioritization. This app applies the paper's comparative-bioinformatics filters and size constraints to those rankings. pFOX is not a calibrated probability of FOX essentiality.

## What the app provides

- Gene search and model-score filtering
- Filamentous and unicellular diazotroph conservation filters
- Non-diazotroph exclusion filters
- Rank-based and probability-per-length complement selection
- Ortholog panels, summary statistics, word clouds, and CSV downloads

## Install and run

```bash
git clone https://github.com/jamesyoung93/FoxGenesApp.git
cd FoxGenesApp
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Stack

Python, Streamlit, pandas, matplotlib, matplotlib-venn, and wordcloud.

## Data availability

The repository includes the candidate table loaded by the app. The table is derived from the public modeling workflow in [FoxGenes_ML](https://github.com/jamesyoung93/FoxGenes_ML). The underlying public data sources and accession identifiers are documented in the paper's Data Availability section.

## Citation

```bibtex
@article{Young2026FoxGenes,
  author  = {Young, James and Gu, Liping and Zhou, Ruanbao},
  title   = {Predicting FOX gene candidates for oxic nitrogen fixation using multi-omic machine learning and comparative bioinformatics},
  journal = {Scientific Reports},
  year    = {2026},
  volume  = {16},
  pages   = {11412},
  doi     = {10.1038/s41598-026-41873-w}
}
```

## License

Code is available under the [MIT License](LICENSE). The bundled candidate table and third-party source data retain their original terms.
