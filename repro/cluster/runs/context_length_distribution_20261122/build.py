"""Reproduce input-prefix length histograms from the frozen prime1 test inputs."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];OUT=ROOT/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
SUITES=[('test100_dfs',100,'100-case DFS test set',REPO/'runs/eval100_official_dfs_20261122/p1_inputs.jsonl'),('test18_original_order',18,'18-case test set (original order)',REPO/'runs/viewer_parse_fox_fix_20261121/p1_inputs.jsonl')]
def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
 report={};bins=np.arange(0,101,10)-.5;centers=(bins[1:]+bins[:-1])/2
 for key,n,title,path in SUITES:
  rows=[json.loads(s) for s in path.open() if s.strip()];assert len(rows)==n*2
  assert len({r['source_key'] for r in rows})==n and len({(r['source_key'],r['condition']) for r in rows})==2*n
  assert all(r['prime']==1 and r['condition'] in ['original','stretched'] for r in rows)
  for r in rows:
   assert sum(' | ' in line for line in r['prefix'].splitlines())==r['n_prefix']
   assert len(r['prefix'].splitlines())==2*r['n_prefix']-1
  a={c:np.array([r['n_prefix'] for r in rows if r['condition']==c]) for c in ['original','stretched']}
  stats={c:{'n':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'min':int(v.min()),'max':int(v.max()),'std':float(v.std(ddof=1))} for c,v in a.items()}
  assert all(len(v)==n and 0<=v.min()<=v.max()<=99 for v in a.values())
  counts={c:np.histogram(v,bins=bins)[0] for c,v in a.items()};assert all(int(v.sum())==n for v in counts.values())
  fig,ax=plt.subplots(figsize=(9,4.9));fig.subplots_adjust(left=.09,right=.98,bottom=.28,top=.80)
  for c,label,color,offset in [('original','Normal','#3678a8',-2.1),('stretched','Stretched','#d27b32',2.1)]:
   heights=counts[c];bars=ax.bar(centers+offset,heights,width=4.0,color=color,label=f'{label} (n = {n})',zorder=3)
   for b,h in zip(bars,heights):
    if h:ax.text(b.get_x()+b.get_width()/2,h+.22,str(h),ha='center',va='bottom',fontsize=9,color=color)
  ymax=max(max(v) for v in counts.values());step=1 if ymax<=7 else 2 if ymax<=14 else 5
  top=int(np.ceil((ymax+2)/step)*step);ax.set_ylim(0,top);ax.set_yticks(np.arange(0,top+1,step));ax.set_xlim(-1,100)
  ax.set_xticks(centers,[f'{i}-{i+9}' for i in range(0,100,10)],fontsize=9)
  ax.set_xlabel('Given context length (number of bricks)',labelpad=9);ax.set_ylabel('Number of test cases');ax.grid(axis='y',alpha=.20,zorder=0);ax.set_axisbelow(True)
  ax.legend(loc='upper right',frameon=False,fontsize=10)
  fig.text(.09,.945,'Given context length distribution',fontsize=17,fontweight='bold',ha='left')
  fig.text(.09,.885,title+' | First target brick provided',fontsize=11,color='#4b5563',ha='left')
  for c,label,y in [('original','Normal',.137),('stretched','Stretched',.095)]:
   s=stats[c];fig.text(.09,y,f'{label}: mean {s["mean"]:.2f}; median {s["median"]:g}; range {s["min"]}-{s["max"]} bricks.',fontsize=10)
  fig.text(.09,.035,'Length = all input bricks, including the first target brick. One count per case and condition; bin width = 10 bricks.',fontsize=8.3,color='#5b6470')
  pdf=OUT/f'given_context_length_{key}.pdf';fig.savefig(pdf);plt.close(fig)
  doc=fitz.open(pdf);assert len(doc)==1;page=doc[0];text=page.get_text();assert 'Given context length distribution' in text and str(n) in text
  page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUT/f'given_context_length_{key}.png');doc.close()
  with (OUT/f'given_context_length_{key}.csv').open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.writer(f);w.writerow(['source_key','condition','given_bricks_including_first_target_brick']);w.writerows([r['source_key'],r['condition'],r['n_prefix']] for r in rows)
  report[key]={'source_file':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'definition':'n_prefix; all given bricks including provided first target brick; not tokens; no replication for four generated samples','stats':stats,'bin_edges':bins.tolist(),'counts':{c:v.tolist() for c,v in counts.items()},'pdf':str(pdf)}
 (OUT/'distribution_statistics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));shutil.copy2(Path(__file__),OUT/'build.py')
 for p in OUT.iterdir():p.chmod(0o775)
 print(json.dumps({k:v['stats'] for k,v in report.items()},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
