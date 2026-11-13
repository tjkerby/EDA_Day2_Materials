from pathlib import Path
import json,sys
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'support'))
from eda_helpers import prepare_titanic,prepare_tornadoes,load_saved,load_variant

payload={}
for name in ['titanic','tornadoes']:
    if name=='titanic':
        df,X,features,_=prepare_titanic(ROOT);ids=df.row_id
    else:
        df,X,features=prepare_tornadoes(ROOT);ids=df.tornado_id
    runs={k:{'label':k,'coords':v.tolist()} for k,v in load_saved(ROOT,name,X,ids).items()}
    if name=='titanic':
        for file,label in [('tsne_perplexity_5','t-SNE: perplexity 5'),
                           ('tsne_seed_7','t-SNE: random initialization, seed 7'),
                           ('umap_neighbors_5','UMAP: 5 neighbors'),('umap_neighbors_50','UMAP: 50 neighbors'),
                           ('umap_min_dist_08','UMAP: min_dist 0.8'),('umap_seed_7','UMAP: seed 7'),
                           ('phate_t_2','PHATE: t=2'),('phate_t_30','PHATE: t=30')]:
            runs[file]={'label':label,'coords':load_variant(ROOT,file,ids).tolist()}
    payload[name]={'records':json.loads(df.to_json(orient='records',date_format='iso')),'features':features,'runs':runs}

template=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Embedding explorer | STAT 386</title><link rel="stylesheet" href="assets/site.css">
<style>
main{max-width:1260px} .controls{display:flex;flex-wrap:wrap;gap:20px;align-items:end;margin:20px 0}
label{display:block;font-weight:600}select,button{font:inherit;padding:8px;max-width:100%}
#plot{width:100%;height:570px} .small{font-size:15px;color:#536c78}.summary{overflow-x:auto}pre{white-space:pre-wrap}
</style></head><body>
<header class="site-header"><nav><a href="index.html">Home</a><a href="slides.html">Slides</a><a href="notebooks.html">Notebooks</a><a href="run_details.html">Run details</a></nav></header>
<main><h1>Embedding explorer</h1>
<p>Keep a question in mind. Change color to investigate a pattern, then select observations and compare their original variables.</p>
<div class="controls">
<div><label for="dataset">Dataset</label><select id="dataset"><option value="titanic">Titanic</option><option value="tornadoes">Tornado practice sample</option></select></div>
<div><label for="run">Representation</label><select id="run"></select></div>
<div><label for="color">Color by</label><select id="color"></select></div>
<button id="clear">Clear selection</button></div>
<p id="context" class="small"></p><div id="plot" aria-label="Interactive scatter plot of saved embedding coordinates"></div>
<p class="small">Drag to select points with the lasso or box tools. Click a point to inspect its record. Changing color keeps coordinates fixed. A selected region is an inspection aid, not a validated cluster.</p>
<h2>Selected observations and remaining records</h2><div id="summary" class="summary" aria-live="polite"></div>
<h2>Record inspection</h2><pre id="record">Click a point to inspect its original variables.</pre>
<p>Next check: explain the region using original variables and compare a reasonable alternative setting. <a href="run_details.html">Saved-run settings and provenance</a></p>
</main>
<script src="assets/plotly.js"></script><script id="embedding-data" type="application/json">__PAYLOAD__</script>
<script>
const data=JSON.parse(document.getElementById('embedding-data').textContent);
const el=id=>document.getElementById(id);
const categories=new Set(['survived','sex','pclass','embarked','age_is_missing','embarked_is_missing','ef_rating','start_state','month']);
const palette=['#087f83','#c76b2a','#7354a3','#719b39','#cb5778','#387aa3','#9e7048','#555555'];
const opts={titanic:['none','survived','sex','pclass','age','fare','age_is_missing','embarked_is_missing','embarked'],tornadoes:['none','ef_rating','month','start_state','path_length_miles','max_width_yards','start_latitude','start_longitude']};
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const fmt=(x,percent=false)=>x===null?'Missing':percent?(100*x).toFixed(1)+'%':x.toLocaleString(undefined,{maximumFractionDigits:2});
const numeric=vals=>vals.filter(x=>x!==null&&Number.isFinite(Number(x))).map(Number);
const mean=vals=>{const x=numeric(vals);return x.length?x.reduce((a,b)=>a+b,0)/x.length:null;};
const median=vals=>{const x=numeric(vals).sort((a,b)=>a-b),n=x.length;return n?(n%2?x[(n-1)/2]:(x[n/2-1]+x[n/2])/2):null;};
function resetOptions(){
 const d=data[el('dataset').value];
 el('run').innerHTML=Object.entries(d.runs).map(([k,v])=>'<option value="'+k+'">'+escape(v.label)+'</option>').join('');el('run').value='UMAP';
 el('color').innerHTML=opts[el('dataset').value].map(x=>'<option>'+x+'</option>').join('');el('color').value=el('dataset').value==='titanic'?'survived':'ef_rating';
 el('context').textContent=d.records.length+' records; '+d.features.length+' embedding features. '+(el('dataset').value==='titanic'?'Survival and missingness indicators are withheld from the inputs.':'Fixed sample from 2015-2018. EF, time, and outcomes are withheld. Starting coordinates are inputs.');
}
function summary(indices){
 const d=data[el('dataset').value],s=new Set(indices),rows=d.records;
 if(!s.size){el('summary').textContent='Select points to compare their original variables with the remaining records.';return;}
 const a=rows.filter((_,i)=>s.has(i)),b=rows.filter((_,i)=>!s.has(i));
 let metrics=[['Records',a.length,b.length,false]];
 if(el('dataset').value==='titanic')metrics.push(['Survival proportion',mean(a.map(r=>r.survived)),mean(b.map(r=>r.survived)),true],['Median fare',median(a.map(r=>r.fare)),median(b.map(r=>r.fare)),false],['Age missing',mean(a.map(r=>r.age_is_missing?1:0)),mean(b.map(r=>r.age_is_missing?1:0)),true]);
 else metrics.push(['Observed EF ratings',a.filter(r=>r.ef_rating!==null).length,b.filter(r=>r.ef_rating!==null).length,false],['Median path length (miles)',median(a.map(r=>r.path_length_miles)),median(b.map(r=>r.path_length_miles)),false],['Median width (yards)',median(a.map(r=>r.max_width_yards)),median(b.map(r=>r.max_width_yards)),false]);
 let h='<table><thead><tr><th>Quantity</th><th>Selected</th><th>Remaining</th></tr></thead><tbody>'+metrics.map(([k,x,y,p])=>'<tr><td>'+escape(k)+'</td><td>'+fmt(x,p)+'</td><td>'+fmt(y,p)+'</td></tr>').join('')+'</tbody></table>';
 const key=el('dataset').value==='titanic'?'pclass':'ef_rating';
 const labels=[...new Set(rows.map(r=>r[key]===null?'Missing':String(r[key])))].sort();
 h+='<table><thead><tr><th>'+key+'</th><th>Selected count</th><th>Remaining count</th></tr></thead><tbody>'+labels.map(l=>'<tr><td>'+escape(l)+'</td><td>'+a.filter(r=>(r[key]===null?'Missing':String(r[key]))===l).length+'</td><td>'+b.filter(r=>(r[key]===null?'Missing':String(r[key]))===l).length+'</td></tr>').join('')+'</tbody></table>';
 el('summary').innerHTML=h;
}
async function draw(){
 const d=data[el('dataset').value],run=d.runs[el('run').value],c=el('color').value,coords=run.coords,rows=d.records;
 const indices=rows.map((_,i)=>i);
 function trace(ids,label,color,continuous=false){return {type:'scattergl',mode:'markers',x:ids.map(i=>coords[i][0]),y:ids.map(i=>coords[i][1]),customdata:ids.map(i=>[i]),text:ids.map(i=>Object.entries(rows[i]).map(([k,v])=>escape(k)+': '+escape(v===null?'Missing':v)).join('<br>')),hovertemplate:'%{text}<extra></extra>',name:label,marker:{size:6,opacity:.75,color,...(continuous?{colorscale:'Viridis',showscale:true,colorbar:{title:c}}:{})}};}
 let traces=[];
 if(c==='none')traces=[trace(indices,'Records','#087f83')];
 else if(categories.has(c)){
  const labels=[...new Set(rows.map(r=>r[c]===null?'Missing':String(r[c])))].sort();
  traces=labels.map((l,j)=>trace(indices.filter(i=>(rows[i][c]===null?'Missing':String(rows[i][c]))===l),l,palette[j%palette.length]));
 }else{
  const valid=indices.filter(i=>rows[i][c]!==null),missing=indices.filter(i=>rows[i][c]===null);
  traces=[trace(valid,c,valid.map(i=>Number(rows[i][c])),true)];
  if(missing.length)traces.push(trace(missing,'Missing','#b9c1c4'));
 }
 await Plotly.react(el('plot'),traces,{title:{text:run.label+' colored by '+c,font:{size:21}},margin:{l:65,r:45,b:60,t:60},xaxis:{title:'Dimension 1'},yaxis:{title:'Dimension 2'},dragmode:'lasso',legend:{title:{text:c}},paper_bgcolor:'#ffffff',plot_bgcolor:'#fbfcfc'},{responsive:true,displaylogo:false});
 el('plot').removeAllListeners('plotly_selected');el('plot').removeAllListeners('plotly_click');
 el('plot').on('plotly_selected',e=>summary(e?e.points.map(p=>p.customdata[0]):[]));
 el('plot').on('plotly_click',e=>{const i=e.points[0].customdata[0];el('record').textContent=JSON.stringify(rows[i],null,2);});
 summary([]);el('record').textContent='Click a point to inspect its original variables.';
}
el('dataset').addEventListener('change',()=>{resetOptions();draw();});
el('run').addEventListener('change',draw);el('color').addEventListener('change',draw);
el('clear').addEventListener('click',()=>{Plotly.restyle(el('plot'),{selectedpoints:[null]});summary([]);});
resetOptions();draw();
</script></body></html>'''
serialized=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
(ROOT/'explorer.html').write_text(template.replace('__PAYLOAD__',serialized))
print('Built the interactive explorer with bundled data and Plotly')
