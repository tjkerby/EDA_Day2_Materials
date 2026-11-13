// Rebuild saved UMAP reference coordinates with Node.js and the bundled official UMAP-js.
// Run from any directory: node reference_tools/rebuild_umap.cjs
const fs=require('fs');
global.window=global;
const {UMAP}=require('./umap-js.min.cjs');
const dir=require('path').join(__dirname,'../data/embeddings');
function seeded(seed){let a=seed;return ()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
for(const dataset of ['titanic','tornadoes']){
  const X=JSON.parse(fs.readFileSync(`${dir}/${dataset}_features.json`));
  const ids=JSON.parse(fs.readFileSync(`${dir}/${dataset}_ids.json`));
  const settings=[['umap',15,0.1,42]];
  if(dataset==='titanic')settings.push(['umap_neighbors_5',5,0.1,42],['umap_neighbors_50',50,0.1,42],['umap_min_dist_08',15,0.8,42],['umap_seed_7',15,0.1,7]);
  for(const [name,nNeighbors,minDist,seed] of settings){
    const start=Date.now();
    const u=new UMAP({nComponents:2,nNeighbors,minDist,nEpochs:400,random:seeded(seed)});
    const Y=u.fit(X);
    if(Y.length!==ids.length||Y.some(r=>r.length!==2||r.some(v=>!Number.isFinite(v))))throw Error('Invalid embedding');
    const rows=['row_id,dim1,dim2',...Y.map((r,i)=>`${ids[i]},${r[0]},${r[1]}`)];
    fs.writeFileSync(`${dir}/${dataset}_${name}.csv`,rows.join('\n')+'\n');
    console.log(dataset,name,Date.now()-start,'ms');
  }
}
