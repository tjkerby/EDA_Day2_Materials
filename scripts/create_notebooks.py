from pathlib import Path
import json,textwrap,uuid

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks'

def md(s):return {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':textwrap.dedent(s).strip()+'\n'}
def code(s):return {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'source':textwrap.dedent(s).strip()+'\n','execution_count':None,'outputs':[]}
def save(name,cells,title):
    obj={'cells':cells,'metadata':{'title':title,'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
    (OUT/name).write_text(json.dumps(obj,indent=1)+'\n')

setup='''
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path.cwd()
if not (ROOT / "support/eda_helpers.py").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "support"))
from eda_helpers import prepare_titanic, prepare_tornadoes, load_saved, load_variant, plot_embedding

# False uses supplied runs and requires only the base environment.
# True recomputes nonlinear methods. First run: uv sync --extra recompute
RECOMPUTE = False
'''

method_cells=[
('PCA: a linear baseline',
'''pca = PCA(n_components=2, svd_solver="full")
embeddings["PCA"] = pca.fit_transform(X)
print("Variance retained:", pca.explained_variance_ratio_.sum())
plot_embedding(embeddings["PCA"], df, title="PCA: passenger profiles")
plt.tight_layout()
plt.show()''',
'''# TODO: Create PCA with two components, then fit_transform(X).
# Replace embeddings["PCA"] with the resulting coordinates.
# Optional: inspect the explained_variance_ratio_.

# The saved run remains available if you skip a computation.
plot_embedding(embeddings["PCA"], df, title="PCA: passenger profiles")
plt.tight_layout()
plt.show()''',
'PCA returns a coordinate for each passenger on each retained component. The first two retain about 53.8% of the variance of these prepared features. Variance is not a measure of relevance to survival.'),
('t-SNE: neighborhood similarities',
'''if RECOMPUTE:
    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                learning_rate="auto", max_iter=1000, random_state=42)
    embeddings["t-SNE"] = tsne.fit_transform(X)

plot_embedding(embeddings["t-SNE"], df, "sex", "t-SNE colored by recorded sex")
plt.tight_layout()
plt.show()''',
'''# TODO: If recomputing, create TSNE and fit_transform(X).
# Suggested settings: perplexity=30, init="pca", learning_rate="auto",
# max_iter=1000, random_state=42.
# Replace embeddings["t-SNE"] with the coordinates.

plot_embedding(embeddings["t-SNE"], df, "sex", "t-SNE colored by recorded sex")
plt.tight_layout()
plt.show()''',
'Nearby points represent a neighborhood-based view of similarity. Inter-island spacing, island area, and point density do not provide faithful measurements of the original feature space.'),
('UMAP: a neighborhood graph',
'''if RECOMPUTE:
    import umap
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="euclidean", random_state=42)
    embeddings["UMAP"] = reducer.fit_transform(X)

plot_embedding(embeddings["UMAP"], df, "pclass", "UMAP colored by passenger class")
plt.tight_layout()
plt.show()''',
'''# TODO: If recomputing, import umap and create umap.UMAP(...).
# Suggested settings: n_neighbors=15, min_dist=0.1, random_state=42.
# Replace embeddings["UMAP"] with fit_transform(X).

plot_embedding(embeddings["UMAP"], df, "pclass", "UMAP colored by passenger class")
plt.tight_layout()
plt.show()''',
'n_neighbors controls the scale of neighborhoods used to build the graph. min_dist influences how tightly the layout packs points. Neither parameter certifies that visible islands are distinct populations.'),
('PHATE: diffusion and broader structure',
'''if RECOMPUTE:
    import phate
    reducer = phate.PHATE(n_components=2, knn=30, t="auto",
                          mds_solver="smacof", random_state=42,
                          n_jobs=1, verbose=0)
    embeddings["PHATE"] = reducer.fit_transform(X)

plot_embedding(embeddings["PHATE"], df, "survived", "PHATE colored by survival")
plt.tight_layout()
plt.show()''',
'''# TODO: If recomputing, import phate and create phate.PHATE(...).
# Suggested settings: knn=30, t="auto", mds_solver="smacof", random_state=42.
# Replace embeddings["PHATE"] with fit_transform(X).

plot_embedding(embeddings["PHATE"], df, "survived", "PHATE colored by survival")
plt.tight_layout()
plt.show()''',
'PHATE compares diffusion profiles on a graph before producing a low-dimensional layout. This can help inspect continuous structure. Diffusion steps are not elapsed time, and a branch does not establish a passenger trajectory. We use knn=30 because repeated feature profiles occur in this dataset, retain all passengers, and add no jitter.')
]

for complete in [True,False]:
    label='Completed reference' if complete else 'Live coding scaffold'
    cells=[md(f'''# Titanic dimensionality reduction: {label}

**Question:** What profiles appear across passenger characteristics, and how do they relate to survival?

{('All application code is supplied. Saved runs make this notebook immediate to use; switch RECOMPUTE to True to fit nonlinear methods yourself.' if complete else 'Preparation and plotting helpers are supplied. Fill in the method calls live. Saved runs are loaded as a fallback, and this scaffold has no saved outputs.')}

The main live path is preparation, one fit_transform call per method, recoloring, and a return to original variables. The SVD and parameter sections are optional follow-up.

**Saved-run provenance:** PCA and t-SNE use scikit-learn, PHATE uses its Python implementation, and UMAP uses the official UMAP-js implementation. The Python UMAP recomputation branch uses umap-learn and was not executable in the authoring environment. The two implementations can produce different layouts. See [run details](../run_details.qmd).'''),
md('## Setup\n\nRun before class. The four saved runs are already computed for the bundled data. Install the recompute extra before using UMAP or PHATE fitting code.'),code(setup),
md('''## Prepared features and plotting metadata

The supplied preparation function imputes numerical fields by the median, standardizes them, and imputes/one-hot encodes categorical fields. It retains all 891 passengers. Open `support/eda_helpers.py` to inspect the full pipeline.

| Embedding inputs | Preparation |
|---|---|
| age, fare, sibsp, parch | Median imputation, then z-scores |
| pclass, sex, embarked | Most-frequent imputation, then one-hot encoding |
| survived | Excluded from inputs, retained for color |
| Missingness indicators | Excluded from inputs, retained for color |

The numerical block and each categorical block contribute differently to Euclidean distances. This is one explicit definition of passenger similarity, not an automatically optimal choice.'''),
code('''df, X, feature_names, preprocessor = prepare_titanic(ROOT)
embeddings = {} if RECOMPUTE else load_saved(ROOT, "titanic", X, df["row_id"])
print("Prepared matrix:", X.shape)
print("Feature columns:", feature_names)
assert "survived" not in feature_names
df[["row_id", "age", "fare", "pclass", "sex", "survived"]].head()''')]
    for title,solution,scaffold,interpretation in method_cells:
        cells.extend([md('## '+title+'\n\nBefore running: what kind of information does this method prioritize?'),code(solution if complete else scaffold),md(('**Interpretation:** '+interpretation) if complete else '**Discuss:** What does this view suggest? What would you color or inspect next?')])
    cells.extend([
md('''## Recolor the same coordinates

Choose a method and a variable, then rerun this cell. Recoloring does not refit the embedding. Compare an input feature, a missingness indicator, and survival, which was withheld.

Options: sex, pclass, age, fare, age_is_missing, embarked_is_missing, survived.'''),
code('''METHOD = "UMAP"
COLOR_BY = "survived"
plot_embedding(embeddings[METHOD], df, COLOR_BY, f"{METHOD}: {COLOR_BY}")
plt.tight_layout()
plt.show()'''),
md('''## A pattern becomes another question

Does apparent survival separation simply reflect passenger class and sex? Return to the original variables and compare proportions with explicit denominators.'''),
code('''comparison = df.groupby(["pclass", "sex"])["survived"].agg(
    passengers="size", survivors="sum", survival_proportion="mean"
)
display(comparison.round(3))
ax = comparison["survival_proportion"].unstack("sex").plot.bar(rot=0)
ax.set(xlabel="Passenger class", ylabel="Survival proportion", ylim=(0, 1))
plt.tight_layout()
plt.show()''' if complete else '''# TODO: Group passengers by class and sex.
# Show group sizes, survivor counts, and survival proportions.
# What does this original-variable view add to the embedding?'''),
md('''## Optional: inspect a region and compare it with the remaining passengers

This example takes the rightmost 20% of one layout as a convenient inspection region. It does not define a validated cluster. Axis direction is arbitrary and the region changes if the embedding rotates or changes settings.'''),
code('''coords = embeddings["UMAP"]
in_region = coords[:, 0] > np.quantile(coords[:, 0], 0.8)
regions = df.assign(region=np.where(in_region, "Selected region", "Remaining passengers"))
display(regions.groupby("region").agg(
    n=("row_id", "size"), survival=("survived", "mean"),
    average_fare=("fare", "mean"), missing_age=("age_is_missing", "mean")
).round(3))
display(pd.crosstab(regions["region"], regions["pclass"], normalize="index").round(3))'''),
md('''## Optional: connect SVD to PCA

For centered X, X = U diag(s) Vt. The first two right singular vectors define directions. U[:, :2] * s[:2] gives point coordinates. A component's sign can flip without changing the represented geometry.'''),
code('''X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
svd_scores = U[:, :2] * s[:2]
variance = s**2 / (len(X) - 1)
print("Variance fraction from SVD:", variance[:2].sum() / variance.sum())
display(pd.DataFrame(Vt[:2].T, index=feature_names, columns=["PC1 weight", "PC2 weight"]).round(3))
print("Scores shape:", svd_scores.shape)'''),
md('''## Optional: a controlled parameter comparison

These supplied runs hold the dataset and preprocessing fixed. The example below compares t-SNE perplexity 5 with 30. Use the same color variable and avoid reading between-island distances as original distances.'''),
code('''alternative = load_variant(ROOT, "tsne_perplexity_5", df["row_id"], X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_embedding(alternative, df, "survived", "Perplexity 5", axes[0])
plot_embedding(load_variant(ROOT, "tsne", df["row_id"], X), df, "survived", "Perplexity 30", axes[1])
plt.tight_layout()
plt.show()'''),
md('''## Optional: UMAP and PHATE sensitivity

Other supplied variants: umap_neighbors_5, umap_neighbors_50, umap_min_dist_08, umap_seed_7, phate_t_2, phate_t_30. The source settings and implementation details appear in the run manifest.

PHATE's t changes diffusion smoothing. It is not a time variable measured on the passengers. Changing settings is a sensitivity check rather than a search for the prettiest plot.

**Next question:** Which interpretation survives a reasonable alternative view and an original-variable check?''')])
    save('titanic_complete.ipynb' if complete else 'titanic_live_scaffold.ipynb',cells,'Titanic DR: '+label)

tornado_setup=setup+'''
df, X, feature_names = prepare_tornadoes(ROOT, include_location=True)
embeddings = {} if RECOMPUTE else load_saved(ROOT, "tornadoes", X, df["tornado_id"])
print("Fixed practice sample:", len(df), "records")
print("Embedding features:", feature_names)
'''
for complete in [True,False]:
    label='Completed reference' if complete else 'Extra practice'
    cells=[md(f'''# Tornado dimensionality reduction: {label}

Optional practice after the main Titanic lesson. The original dataset has 4,706 whole tracks from 2015-2018. This exercise uses a fixed 700-record sample for speed.

**Question:** What profiles appear across track size and starting location, and how do those profiles relate to EF rating?

The inputs are standardized log1p length, log1p width, starting latitude, and starting longitude. EF rating, month, start-state labels, injuries, and fatalities are withheld. Longitude/latitude here are exploratory features, not a precise physical-distance model.

All four saved methods are supplied. The same UMAP-js versus Python umap-learn provenance note applies as in Titanic. See [run details](../run_details.qmd).'''),
md('## Setup and prepared features\n\nKeep missing EF ratings as missing. They do not prevent fitting these four complete input features.'),code(tornado_setup),
md('## A familiar starting view\n\nCompare size variables directly before interpreting their multivariable embedding. The transformations reduce domination by very large recorded tracks.'),
code('''plt.scatter(np.log1p(df["path_length_miles"]), np.log1p(df["max_width_yards"]), s=15, alpha=0.6)
plt.xlabel("log(1 + path length in miles)")
plt.ylabel("log(1 + maximum width in yards)")
plt.tight_layout()
plt.show()''')]
    for title,solution,scaffold,interpretation in method_cells:
        sol=solution.replace('"sex"','"ef_rating"').replace('"pclass"','"ef_rating"').replace('"survived"','"ef_rating"').replace('passenger profiles','tornado profiles').replace('recorded sex','EF rating').replace('passenger class','EF rating').replace('survival','EF rating')
        if complete:cells.extend([md('## '+title),code(sol)])
        else:cells.extend([md('## '+title+'\n\nApply the method using the Titanic notebook as a reference, or explore the supplied coordinates.'),code('''# TODO: Fit this method and replace its entry in embeddings.
# The supplied entry remains available for exploration.
''')])
    cells.extend([md('''## Investigate a question using color

Try ef_rating, month, path_length_miles, max_width_yards, start_state, or start_latitude. Which overlays describe features already used to construct the geometry? Which supply withheld information?'''),
code('''METHOD = "PHATE"
COLOR_BY = "ef_rating"
plot_embedding(embeddings[METHOD], df, COLOR_BY, f"{METHOD}: {COLOR_BY}")
plt.tight_layout()
plt.show()'''),
md('''## Return to original variables

Inspect the number of observed ratings before comparing sizes across EF categories. This is a descriptive comparison within the practice sample, not a population risk estimate.'''),
code('''rated = df.dropna(subset=["ef_rating"])
print(f"Observed EF rating: {len(rated)} of {len(df)} sample records")
display(rated.groupby("ef_rating").agg(
    n=("tornado_id", "size"),
    median_length_miles=("path_length_miles", "median"),
    median_width_yards=("max_width_yards", "median")
).round(2))''' if complete else '''# TODO: Choose a relevant original-variable comparison.
# Count the records used and explain any missing-value exclusions.'''),
md('''## Further practice

- Compare a size-only PCA with the size-and-location PCA. `prepare_tornadoes(ROOT, include_location=False)` provides the two size inputs. Why is a direct two-variable scatter already informative in that case?
- Explain how including geography influences the meaning of a visible grouping.
- Identify a question this dataset cannot answer without additional information.

Saved coordinates are tied to the supplied feature matrix. Changed preparation requires recomputation, not reuse of the old cache.''')])
    save('tornado_complete.ipynb' if complete else 'tornado_practice.ipynb',cells,'Tornado DR: '+label)
print('Created four Day 2 notebooks')
