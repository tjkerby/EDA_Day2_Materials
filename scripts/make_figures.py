from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'support'))
from eda_helpers import prepare_titanic,prepare_tornadoes,load_saved,load_variant,plot_embedding
OUT=ROOT/'assets';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':15,
 'axes.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close()

rng=np.random.default_rng(386)
cloud=rng.multivariate_normal([0,0],[[3,2.4],[2.4,2.5]],size=90)
pca=PCA().fit(cloud)
v=pca.components_[0]
projection=(cloud@v)[:,None]*v[None,:]
fig,axs=plt.subplots(1,2,figsize=(11,4.1))
for ax in axs:
    ax.scatter(cloud[:,0],cloud[:,1],s=20,color='#78949e',alpha=.7)
    ax.set(xlim=(-5.5,5.5),ylim=(-5.5,5.5),xlabel='Feature 1',ylabel='Feature 2',aspect='equal')
    ax.axhline(0,c='#d9e1e4',lw=.8);ax.axvline(0,c='#d9e1e4',lw=.8)
for i,(direction,color,label) in enumerate(zip(pca.components_,['#087f83','#c76b2a'],['PC1','PC2'])):
    axs[0].arrow(0,0,*direction*4,width=.035,color=color,length_includes_head=True)
    axs[0].text(*(direction*4.4),label,color=color,fontsize=13)
axs[0].set_title('New perpendicular directions')
for i in range(0,len(cloud),6):
    axs[1].plot([cloud[i,0],projection[i,0]],[cloud[i,1],projection[i,1]],c='#b9c5ca',lw=.8)
axs[1].scatter(projection[:,0],projection[:,1],s=16,color='#087f83')
axs[1].plot([-5*v[0],5*v[0]],[-5*v[1],5*v[1]],c='#087f83')
axs[1].set_title('Projection retains one coordinate')
save('pca_geometry')

scaled_units=cloud*np.array([100,1])
standardized=StandardScaler().fit_transform(scaled_units)
fig,axs=plt.subplots(1,2,figsize=(10.8,3.8))
for ax,data,title in zip(axs,[scaled_units,standardized],['One feature uses 100 times larger units','After standardizing both features']):
    ax.scatter(data[:,0],data[:,1],s=18,color='#087f83',alpha=.65)
    pc=PCA().fit(data)
    direction=pc.components_[0]
    length=np.std(data@direction)*2.4
    ax.plot([-length*direction[0],length*direction[0]],[-length*direction[1],length*direction[1]],color='#c76b2a',lw=3,label='PC1 direction')
    ax.set(xlabel='Feature 1',ylabel='Feature 2',title=title)
    ax.legend(frameon=False,fontsize=10)
save('scaling')

df,X,names,_=prepare_titanic(ROOT)
emb=load_saved(ROOT,'titanic',X,df.row_id)
pca=PCA().fit(X)
fig,axs=plt.subplots(1,2,figsize=(10.5,3.9))
axs[0].bar(np.arange(1,len(names)+1),pca.explained_variance_ratio_,color='#087f83')
axs[0].set(xlabel='Principal component',ylabel='Variance fraction',xticks=np.arange(1,len(names)+1))
axs[1].plot(np.arange(1,len(names)+1),np.cumsum(pca.explained_variance_ratio_),marker='o',color='#087f83')
axs[1].axvline(2,c='#c76b2a',ls='--')
axs[1].set(xlabel='Components retained',ylabel='Cumulative variance fraction',ylim=(0,1.06))
axs[1].text(2.3,.49,f'First two: {pca.explained_variance_ratio_[:2].sum():.1%}',fontsize=14,color='#c76b2a')
save('pca_variance')

fig,ax=plt.subplots(figsize=(11.5,3.5))
w=pca.components_[:2].T
im=ax.imshow(w.T,cmap='RdBu_r',vmin=-.8,vmax=.8,aspect='auto')
ax.set(yticks=[0,1],yticklabels=['PC1','PC2'],xticks=np.arange(len(names)),xticklabels=names)
plt.setp(ax.get_xticklabels(),rotation=35,ha='right',fontsize=16)
plt.setp(ax.get_yticklabels(),fontsize=17)
for i in range(len(names)):
    for j in range(2):ax.text(i,j,f'{w[i,j]:.2f}',ha='center',va='center',fontsize=13,color='white' if abs(w[i,j])>.5 else '#182d3b')
fig.colorbar(im,ax=ax,label='Component weight')
save('pca_weights')

fig,axs=plt.subplots(2,2,figsize=(11,7))
for ax,method in zip(axs.flat,emb):plot_embedding(emb[method],df,'survived',method,ax)
save('four_methods')
fig,axs=plt.subplots(1,2,figsize=(11,3.6))
for ax,field in zip(axs.flat,['sex','survived']):
    plot_embedding(emb['UMAP'],df,field,'Same UMAP coordinates: '+field,ax)
    for t in ax.get_legend().get_texts():t.set_fontsize(13)
save('color_overlays')

for name,title,items in [
    ('tsne_perplexity','t-SNE', [('Perplexity 5',load_variant(ROOT,'tsne_perplexity_5',df.row_id)),('Perplexity 30',emb['t-SNE'])]),
    ('umap_neighbors','UMAP', [('5 neighbors',load_variant(ROOT,'umap_neighbors_5',df.row_id)),('15 neighbors',emb['UMAP']),('50 neighbors',load_variant(ROOT,'umap_neighbors_50',df.row_id))]),
    ('umap_min_dist','UMAP', [('min_dist = 0.1',emb['UMAP']),('min_dist = 0.8',load_variant(ROOT,'umap_min_dist_08',df.row_id))]),
    ('phate_diffusion','PHATE', [('t = 2',load_variant(ROOT,'phate_t_2',df.row_id)),('t = auto',emb['PHATE']),('t = 30',load_variant(ROOT,'phate_t_30',df.row_id))]),
]:
    fig,axs=plt.subplots(1,len(items),figsize=(4*len(items),3.6))
    for ax,(label,coords) in zip(np.atleast_1d(axs),items):plot_embedding(coords,df,'survived',label,ax)
    save(name)

points=np.array([[0,0],[1,.4],[1.6,-.2],[2.6,.2],[3.4,-.2],[4.2,.2],[5,.0],[5.8,.5],[3.4,1.1],[3.5,2.1],[4.3,2.7],[5.2,2.8]])
G=kneighbors_graph(points,3,mode='distance').toarray()
fig,ax=plt.subplots(figsize=(8,3.4))
for i in range(len(points)):
    for j in range(i+1,len(points)):
        if G[i,j]>0 or G[j,i]>0:ax.plot(points[[i,j],0],points[[i,j],1],c='#a6b7be',lw=1)
ax.scatter(points[:,0],points[:,1],c='#087f83',s=75)
ax.set(title='Illustration: 3-nearest-neighbor connections',xlabel='Feature 1',ylabel='Feature 2')
save('neighbor_graph')

A=((G>0)|(G.T>0)).astype(float)+np.eye(len(points))
P=A/A.sum(axis=1,keepdims=True)
probs=np.stack([np.linalg.matrix_power(P,t)[0] for t in [1,3,10]])
fig,ax=plt.subplots(figsize=(9.5,2.5))
im=ax.imshow(probs,cmap='YlGnBu',aspect='auto',vmin=0,vmax=probs.max())
ax.set(yticks=[0,1,2],yticklabels=['1 step','3 steps','10 steps'],xticks=np.arange(len(points)),xticklabels=np.arange(len(points)),xlabel='Graph node',title='Illustration: a walk spreads probability from node 0')
fig.colorbar(im,ax=ax,label='Probability')
save('diffusion_intuition')

tdf,TX,tn=prepare_tornadoes(ROOT)
temb=load_saved(ROOT,'tornadoes',TX,tdf.tornado_id)
fig,axs=plt.subplots(2,2,figsize=(11,7))
for ax,method in zip(axs.flat,temb):plot_embedding(temb[method],tdf,'ef_rating',method,ax)
save('tornado_methods')
print('Created teaching figures')
