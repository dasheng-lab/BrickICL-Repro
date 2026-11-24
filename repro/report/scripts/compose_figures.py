"""Compose vector report diagrams and labeled panels from actual archived Viewer renders."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Circle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];IMG=ROOT/'assets/renders';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
COL={'ink':'#19314a','body':'#aeb6bf','ref':'#efb648','target':'#41b7a7','blue':'#367ca8','orange':'#d77932'}
def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',pad_inches=.03);fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.03);plt.close(fig)
def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
 fig=plt.figure(figsize=(10.3,5.45),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1360),ylim=(0,720));ax.axis('off')
 def box(x,y,w,h,title,number,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=13',facecolor='#f8fafc',edgecolor='#bfcbd6',linewidth=1));ax.add_patch(Circle((x+27,y+h-31),16,facecolor=color,edgecolor='none'));ax.text(x+27,y+h-31,str(number),ha='center',va='center',color='white',fontsize=13,fontweight='bold');ax.text(x+55,y+h-32,title,va='center',fontsize=17,fontweight='bold',color=COL['ink'])
 def txt(x,y,s,size=14,**kw):ax.text(x,y,s,fontsize=size,color=COL['ink'],**kw)
 def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.2,color='#718396'))
 box(8,355,420,355,'Data and structure',1,COL['blue']);box(458,355,425,355,'Training recipes',2,COL['orange']);box(913,355,439,355,'Model adaptation',3,COL['target'])
 txt(30,638,'Separate train / validation sources',12)
 for x,name in [(24,'pipeline_fox_original'),(222,'pipeline_fox_stretched')]:
  ia=fig.add_axes([x/1360,430/720,190/1360,205/720]);ia.imshow(Image.open(IMG/(name+'.png')));ia.axis('off')
 txt(110,418,'Normal',14,ha='center');txt(315,418,'Extended',14,ha='center')
 for x,label,color in [(30,'B: body',COL['body']),(150,'R: ref.',COL['ref']),(282,'M: target',COL['target'])]:ax.add_patch(Rectangle((x,383),14,14,color=color));txt(x+20,383,label,12.5)
 txt(480,630,'Training sources only',13)
 def tokens(y,missing=False):
  labels=([('B',COL['body'],110),('R',COL['ref'],90),('M',COL['target'],100),('EOS','#d7c9e8',50)] if missing else [('B',COL['body'],70),('R',COL['ref'],40),('B',COL['body'],60),('M',COL['target'],45),('R',COL['ref'],40),('M',COL['target'],45),('EOS','#d7c9e8',50)]);x=482
  for lab,col,w in labels:
   ax.add_patch(Rectangle((x,y),w,35,facecolor=col,edgecolor='white',linewidth=1));txt(x+w/2,y+17,lab,13,ha='center',va='center')
   if missing and lab in ['B','R']:ax.add_patch(Rectangle((x,y),w,35,facecolor='white',alpha=.55,hatch='///',edgecolor='#b6c0c8',linewidth=0))
   x+=w
 txt(480,593,'M-only',15,fontweight='bold');tokens(550,True);txt(480,524,'Loss: target + EOS',13)
 txt(480,485,'Full-G',15,fontweight='bold');tokens(438,False);txt(480,412,'Loss: full DFS build',13)
 txt(480,383,'Paired DFS / subtree-last control',12)
 txt(936,626,'BrickNet PT / raw Qwen3',15,fontweight='bold');txt(936,592,'0.6B base + fresh LoRA',14)
 txt(936,550,'Regularization',15,fontweight='bold');txt(936,520,'Weight decay + dropout',14)
 txt(936,480,'Structural RL',15,fontweight='bold');txt(936,448,'Subtree-last PT + Pose F2',13.5)
 txt(936,401,'Choose checkpoint on dev data',13)
 txt(936,371,'Keep test inputs fixed',13)
 arrow((431,530),(455,530));arrow((887,530),(910,530))
 box(8,10,660,307,'Conditional completion',4,COL['blue']);box(699,10,653,307,'Independent generation',5,COL['target'])
 txt(31,232,'B + reference + first target brick',16,fontweight='bold');txt(31,197,'100 validation-derived cases · DFS',14);txt(31,163,'Normal / extended · 4 draws each',14)
 for x,label,color in [(35,'B',COL['body']),(102,'R',COL['ref']),(169,'head',COL['target'])]:ax.add_patch(FancyBboxPatch((x,104),60,33,boxstyle='round,pad=0,rounding_size=4',facecolor=color,edgecolor='none'));txt(x+30,121,label,12,ha='center',va='center')
 arrow((239,121),(280,121));txt(296,121,'model → remaining M',14,va='center')
 txt(31,71,'Pose F1: parts and placement',14);txt(31,34,'Proposal DLO: reference response',14)
 txt(721,232,'a only; no LEGO reference',16,fontweight='bold');txt(721,197,'64 draws per stopping policy',14);txt(721,163,'Allow EOS / suppress EOS',14)
 txt(727,121,'a',20,fontweight='bold',va='center');arrow((753,121),(792,121));txt(807,121,'model → parts → stop',14,va='center')
 txt(721,71,'Valid length: before parse failure',14);txt(721,34,'Collision-free length: before either failure',13)
 arrow((1127,352),(1127,322));arrow((1040,340),(344,340));arrow((344,340),(344,321));save(fig,'pipeline')
 prov={p['scene']:p for p in json.loads((ROOT/'assets/render_provenance.json').read_text(encoding='utf-8'))}
 fig,axes=plt.subplots(2,4,figsize=(7.2,5.0));fig.subplots_adjust(left=.005,right=.995,bottom=.06,top=.91,wspace=.035,hspace=.33)
 for ri,(case,label) in enumerate([('fox','Fox #114 · draw 3'),('spacecraft','Spacecraft #497 · draw 1')]):
  for ci,(arm,title) in enumerate([('target','Target reference'),('frozen_pt','B0: released PT'),('post_pt','B-M: M-only'),('grpo_pt','B-RL: GRPO')]):
   name=case+'_'+arm;ax=axes[ri,ci];ax.imshow(Image.open(IMG/(name+'.png')));ax.axis('off');ax.set_title(title,fontsize=10.5,fontweight='bold',pad=5)
   if arm=='target':detail='Ground truth'
   else:
    m=prov[name]['source']['metrics'];detail=f"F1 {m['pose_f1']:.2f}; added {m['n_generated']}/{m['target_parts']}";detail+='\n'+('Collision-free' if m['generation_collision_free'] else 'Collision / parse failure')
   ax.text(.5,-.03,detail,transform=ax.transAxes,ha='center',va='top',fontsize=9.2,color='#374151')
  axes[ri,0].text(0,1.25,label,transform=axes[ri,0].transAxes,fontsize=10.5,color=COL['blue'],fontweight='bold')
 save(fig,'qualitative')
 fig,axes=plt.subplots(2,4,figsize=(7.2,4.4));fig.subplots_adjust(left=.005,right=.995,bottom=.025,top=.93,wspace=.025,hspace=.12)
 for col,sid in enumerate([20,489,499,417]):
  for row,cond in enumerate(['original','stretched']):
   ax=axes[row,col];ax.imshow(Image.open(IMG/f'dataset_{sid}_{cond}.png'));ax.axis('off');ax.set_title(f'#{sid} · '+('normal' if row==0 else 'extended'),fontsize=10)
 save(fig,'dataset_examples');print('Composed pipeline, actual-output comparison and dataset examples')
if __name__=='__main__':main()
