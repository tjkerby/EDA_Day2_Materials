# STAT 386: EDA Day 2

A 50-minute lesson on PCA, t-SNE, UMAP, and PHATE, followed by optional tornado practice. One Quarto website project contains the slides, completed code, live scaffolds, saved runs, an interactive explorer, and teaching notes. Rendered pages live in **docs/** for GitHub Pages.

## Start here

- Open **docs/index.html** to browse the supplied static website.
- Open **slides.qmd** with Quarto preview to teach or edit the Reveal.js slides.
- Open **notebooks/titanic_live_scaffold.ipynb** in Positron and keep **titanic_complete.ipynb** in a second tab.
- Use **explorer.html** or **docs/explorer.html** for immediate recoloring, hover inspection, and lasso/box selection.
- Optional practice: **tornado_practice.ipynb** and **tornado_complete.ipynb**.

The full project is standalone. It uses the same data snapshots as Day 1 and does not modify the Day 1 package.

## Environment

Extract the whole folder and open that folder in Positron. In the project terminal:

```bash
uv sync
```

Select the `.venv` Python interpreter as the notebook kernel. The base environment supports all supplied saved results, original-variable checks, and fresh PCA/t-SNE fitting. No data download is needed after setup.

To fit UMAP and PHATE yourself:

```bash
uv sync --extra recompute
```

Set **RECOMPUTE = True** in the completed notebook. Run it once before class, especially to warm up Python UMAP's first compilation. The live scaffold leaves the application calls for you to fill in, with saved coordinates already available as a fallback. It has no saved outputs. It does not refit merely because you change the color variable.

Optional Plotly use in your own Python code can be enabled with `uv sync --extra interactive`. The supplied browser explorer already includes its own local Plotly runtime.

## Files

| File | Purpose |
|---|---|
| `slides.qmd` | Quarto Reveal.js lecture, with notes and optional appendices |
| `notebooks/titanic_complete.ipynb` | Four method applications, overlays, original-variable checks, optional SVD and sensitivity sections |
| `notebooks/titanic_live_scaffold.ipynb` | Matching live-teaching scaffold with preparation and plotting supplied |
| `notebooks/tornado_practice.ipynb` | Optional student exercise on a fixed 700-record sample |
| `notebooks/tornado_complete.ipynb` | Completed application of all four methods to that sample |
| `explorer.html` | Method/color selectors, point hover, lasso/box selection, selected-versus-remaining comparisons |
| `support/eda_helpers.py` | Preparation, plotting and fingerprint-checked cache loading |
| `run_details.qmd` | Feature definitions, exact main settings, provenance, and verification limits |
| `teaching_notes.qmd` | Timing, conceptual checks, and suggestions for live teaching |

## Fifty minutes

| Clock | Focus |
|---|---|
| 0-3 | Review Day 1 and motivate multivariable profiles |
| 3-17 | PCA geometry, SVD, scores, weights, explained variance, scaling |
| 17-24 | t-SNE similarities and perplexity |
| 24-30 | UMAP graph and parameter interpretation |
| 30-34 | PHATE diffusion intuition, without a derivation |
| 34-46 | Live Titanic exploration |
| 46-50 | Method comparison and interpretation check |

The tornado exercise, covariance derivation and sensitivity appendices are optional. Keep the main live code short and protect the time spent interpreting colors and checking original variables.

## Data and saved results

Titanic uses 891 records and 12 prepared features. Survival and missingness indicators are withheld from the feature matrix. Numerical inputs are imputed and standardized, while categories use imputation and one-hot encoding. The first two PCs retain 53.8% of prepared-feature variance.

The tornado source has 4,706 whole tracks from 2015-2018. The fixed practice sample has 700 records. Its inputs are standardized log1p length, log1p width, starting latitude, and starting longitude. EF, time and outcomes are withheld. This is an exploratory sample, not a population risk estimate.

**Implementation note:** Saved PCA/t-SNE were computed with scikit-learn 1.8.0. PHATE was computed with upstream Python PHATE 2.0.0 and graphtools 2.1.0 source. Saved UMAP was computed with official UMAP-js because Python UMAP's compiled dependencies were unavailable here. The supplied Python code uses `umap-learn`; that recompute branch was not executed in the authoring environment and may produce different layouts. All saved coordinates come from actual method computations, and their settings, source identity and feature fingerprints are documented.

Repeated passenger feature profiles remain in the data. PHATE uses knn=30, no jitter, and no removal of people. PHATE can warn about zero distances between repeated profiles. The supplied runs completed with finite coordinates.

The notebook cells were checked in saved-run mode. PCA/t-SNE and PHATE fitting were also run during coordinate preparation. Changing preparation or row order requires fresh fitting rather than loading an incompatible cache.

## Quarto and GitHub Pages

Install the Quarto CLI separately from the Python packages. Preview or rebuild with:

```bash
quarto preview
quarto render
```

The project writes to **docs/**. It renders saved notebook outputs without rerunning analyses, packages a combined notebook-and-data ZIP, and creates `.nojekyll`.

To publish the supplied site, place this folder's contents at your repository root, commit `docs/`, and configure **Settings > Pages > Deploy from a branch**, choosing your branch and **/docs**. Paths are relative, so no owner/repository URL needs editing. After changes, render and commit the updated source and `docs/`.

Quarto was unavailable in the authoring environment. The supplied docs site is a prebuilt static preview generated from the same Quarto sources with Pandoc. A direct Quarto render has not been run here. The optional manual **Render Quarto and check links** workflow lets you render on GitHub's runner and download the resulting docs artifact before publishing changes.

The fallback authoring command is `python scripts/build_static_preview.py`, requiring Pandoc, NumPy, matplotlib, and PyYAML. Normal maintenance can simply use Quarto. The source `assets/` contains the prepared teaching charts, so normal Quarto rendering does not fit models or regenerate them.

## Updating material

Edit the `.qmd` pages directly and edit notebook copies in Positron. `scripts/create_notebooks.py` regenerates notebook sources and overwrites direct edits, so use it only when you intentionally want a full regeneration. The verification helper preserves a clean live scaffold while retaining completed outputs.

The local source snapshots and `scripts/prepare_data.py` recreate the teaching CSVs. The saved-run manifests and original feature matrices allow the coordinates to be traced to a specific preparation.
