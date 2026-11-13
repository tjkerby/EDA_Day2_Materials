from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'downloads';out.mkdir(exist_ok=True)
for notebook in (ROOT/'notebooks').glob('*.ipynb'):
    shutil.copy2(notebook,out/notebook.name)
with ZipFile(out/'EDA_Day2_Notebooks_and_Data.zip','w',ZIP_DEFLATED) as z:
    for pattern in ['notebooks/*.ipynb','data/*.csv','data/*.md','data/embeddings/*','support/*.py','pyproject.toml']:
        for p in ROOT.glob(pattern):
            if p.is_file():z.write(p,Path('EDA_Day2')/p.relative_to(ROOT))
    z.writestr('EDA_Day2/README.md','''# EDA Day 2 notebooks

Extract this whole folder and open it in Positron. Keep data/, support/, and notebooks/ together.

Run uv sync, then select the .venv Python interpreter as the notebook kernel.
The saved examples work with the base environment and local data.

For fresh Python UMAP/PHATE fits: uv sync --extra recompute.
Set RECOMPUTE = True in a completed notebook, or fill in the live scaffold's fitting calls.

Titanic: titanic_complete.ipynb and titanic_live_scaffold.ipynb.
Optional tornado exercise: tornado_practice.ipynb and tornado_complete.ipynb.

Saved UMAP uses official UMAP-js. Python recompute uses umap-learn and can produce a different layout.
Saved PCA/t-SNE use scikit-learn; PHATE uses its Python implementation.
Feature fingerprints and row IDs are validated before saved runs load.
''')
print('Prepared the combined notebook/data download')
