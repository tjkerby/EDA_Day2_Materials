"""Small preparation and plotting helpers for the Day 2 teaching notebooks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METHODS = ['PCA', 't-SNE', 'UMAP', 'PHATE']

def project_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / 'data/titanic.csv').exists() and (p / 'support').exists():
            return p
    raise FileNotFoundError('Open the Day 2 project folder in Positron first.')

def prepare_titanic(root):
    df = pd.read_csv(Path(root) / 'data/titanic.csv')
    df['age_is_missing'] = df.age.isna()
    df['embarked_is_missing'] = df.embarked.isna()
    numeric = ['age', 'fare', 'sibsp', 'parch']
    categorical = ['pclass', 'sex', 'embarked']
    pre = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]), categorical)
    ], verbose_feature_names_out=False)
    X = np.asarray(pre.fit_transform(df), dtype=float)
    return df, X, list(pre.get_feature_names_out()), pre

def prepare_tornadoes(root, include_location=True):
    df = pd.read_csv(Path(root) / 'data/tornadoes.csv', parse_dates=['event_date'])
    df['ef_rating'] = df.ef_rating.astype('Int64')
    # Sample for an interactive exercise, not for estimating population rates.
    df = df.sample(n=min(700, len(df)), random_state=386).sort_values('tornado_id').reset_index(drop=True)
    # Time, EF rating, and outcomes stay outside the embedding.
    features = pd.DataFrame({
        'log1p_length': np.log1p(df.path_length_miles),
        'log1p_width': np.log1p(df.max_width_yards),
    })
    if include_location:
        features['start_latitude'] = df.start_latitude
        features['start_longitude'] = df.start_longitude
    X = StandardScaler().fit_transform(features)
    return df, np.asarray(X), list(features.columns)

def matrix_hash(X):
    rounded = np.ascontiguousarray(np.round(X, 10), dtype='<f8')
    return hashlib.sha256(rounded.tobytes()).hexdigest()

def load_saved(root, dataset, X, row_ids):
    directory = Path(root) / 'data/embeddings'
    meta = json.loads((directory / f'{dataset}_manifest.json').read_text())
    if meta['matrix_sha256_rounded_10dp'] != matrix_hash(X):
        raise ValueError('Prepared features differ from the saved run. Recompute for your changed preparation.')
    result = {}
    for method in METHODS:
        name = method.lower().replace('-', '')
        table = pd.read_csv(directory / f'{dataset}_{name}.csv', dtype={'row_id':'string'})
        if table.row_id.tolist() != pd.Series(row_ids).astype('string').tolist():
            raise ValueError('Saved embedding row IDs do not match this dataset/order.')
        result[method] = table[['dim1','dim2']].to_numpy()
    return result

def plot_embedding(coords, data, color_by=None, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8,5))
    coords = np.asarray(coords)
    if color_by is None:
        ax.scatter(coords[:,0], coords[:,1], s=15, alpha=.7, color='#087f83')
    else:
        s = data[color_by]
        categorical = color_by in ['survived','pclass','sex','embarked','age_is_missing',
                                   'embarked_is_missing','ef_rating','start_state','month']
        if categorical or not pd.api.types.is_numeric_dtype(s):
            labels = s.astype('string').fillna('Missing')
            for i, label in enumerate(sorted(labels.unique())):
                keep = labels.eq(label).to_numpy(dtype=bool)
                ax.scatter(coords[keep,0], coords[keep,1], s=15, alpha=.7,
                           label=str(label), color=plt.get_cmap('tab10')(i%10))
            ax.legend(title=color_by, frameon=False, fontsize=9, title_fontsize=10,
                      loc='best', markerscale=1.5)
        else:
            valid = s.notna().to_numpy()
            m = ax.scatter(coords[valid,0],coords[valid,1],c=s[valid],s=15,alpha=.8,cmap='viridis')
            if (~valid).any():
                ax.scatter(coords[~valid,0],coords[~valid,1],color='#bbbbbb',s=15,label='Missing')
                ax.legend(frameon=False,fontsize=9)
            plt.colorbar(m, ax=ax, label=color_by)
    ax.set(xlabel='Dimension 1', ylabel='Dimension 2', title=title or '')
    return ax

def fit_phate(X, seed=42, t='auto'):
    import phate
    # A larger neighborhood avoids zero bandwidth for repeated passenger profiles.
    # No records are removed, and no jitter is added.
    return phate.PHATE(n_components=2, knn=30, t=t, mds_solver='smacof',
                       random_state=seed, n_jobs=1, verbose=0).fit_transform(X)

def load_variant(root, name, row_ids, X=None):
    if X is not None:
        meta = json.loads((Path(root) / 'data/embeddings/titanic_manifest.json').read_text())
        if meta['matrix_sha256_rounded_10dp'] != matrix_hash(X):
            raise ValueError('Prepared features differ from these parameter runs. Fit a new comparison.')
    table = pd.read_csv(Path(root) / 'data/embeddings' / f'titanic_{name}.csv', dtype={'row_id':'string'})
    if table.row_id.tolist() != pd.Series(row_ids).astype('string').tolist():
        raise ValueError('Variant row IDs do not match.')
    return table[['dim1','dim2']].to_numpy()
