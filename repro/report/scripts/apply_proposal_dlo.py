"""Publish the proposal's teacher-forced DLO without relabeling sampled statistics."""
import csv, json, re, shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/proposal_dlo_100'
OUT = ROOT / 'generated/proposal_dlo_outputs'
OUT.mkdir(parents=True, exist_ok=True)
summary = {r['run']:r for r in csv.DictReader((INPUT/'summary.csv').open())}
phys = json.loads((ROOT/'data/physical.json').read_text(encoding='utf-8'))
stats = json.loads((ROOT/'generated/statistics.json').read_text(encoding='utf-8'))['models']
names = {'frozen_pt':'B0','post_pt':'B-M','fullg_pt':'B-G','plain':'B-DFS','masked':'B-LATE','direct_m':'Q-M','direct_g':'Q-G','nzd_v2_plain':'Q-DFS','nzd_v2_masked':'Q-LATE','regularized_g':'Q-G-R1','regularized_g_wd010_do015':'Q-G-R2','grpo_pt':'B-RL'}
assert set(summary)==set(names)
shutil.copy2(INPUT/'summary.csv',OUT/'proposal_dlo_100_summary.csv')
shutil.copy2(INPUT/'contrasts.csv',OUT/'proposal_dlo_100_contrasts.csv')

def patch_table(path, old_idx, new_idx):
    lines=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        fields=line.split(' & ')
        if len(fields)<=new_idx:continue
        key=next((k for k,v in names.items() if v==fields[0].strip()),None)
        if not key:continue
        if old_idx is not None:fields[old_idx]='--'
        fields[new_idx]=f"{float(summary[key]['mean_dlo']):+.3f}"
        lines[i]=' & '.join(fields)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

patch_table(ROOT/'tables/main.tex',None,2)
patch_table(ROOT/'tables/all_suites.tex',1,4)
for name in ['sft','confidence_intervals']:
    p=ROOT/'tables'/f'{name}.tex'
    lines=p.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        f=line.split(' & ')
        if len(f)<3:continue
        key=next((k for k,v in names.items() if v==f[0].strip()),None)
        if name=='sft' and f[0].strip().startswith(('Released SFT','SFT +')):f[1]='--'
        elif key and name=='confidence_intervals':
            r=summary[key];f[1]=f"{float(r['mean_dlo']):.3f} [{float(r['lower95']):.3f}, {float(r['upper95']):.3f}]"
        else:continue
        lines[i]=' & '.join(f)
    p.write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Historical New100 checkpoints were not rescored; leave their DLO cells blank.
p=ROOT/'tables/all_suites.tex';s=p.read_text(encoding='utf-8')
for label in ['Q-official70k','Q-v4-DFS','Q-v4-last']:
    s=re.sub(r'(^'+re.escape(label)+r' & [^&]+ & [^&]+ & [^&]+ & )[^&]+( & )',r'\1--\2',s,flags=re.M)
s=s.replace('ID & $\\Delta\\ell$', 'ID & $\\Delta L$')
p.write_text(s,encoding='utf-8')

p=ROOT/'tables/main.tex';s=p.read_text(encoding='utf-8');s=s.replace('Q-M & 750 & +59.222','Q-M & 750 & \\textbf{+59.222}');p.write_text(s,encoding='utf-8')

# Keep generated numerical artifacts consistent with the corrected table.
p=ROOT/'generated/statistics.json';g=json.loads(p.read_text(encoding='utf-8'))
for arm,r in summary.items():
    v=g['models'][arm];v['completion_mean'][0]=float(r['mean_dlo']);v['completion_ci95'][0]=[float(r['lower95']),float(r['upper95'])]
g['proposal_dlo_source']='data/proposal_dlo_100/results/scores/*.csv; complete teacher-forced target including EOS'
p.write_text(json.dumps(g,indent=2),encoding='utf-8')
p=ROOT/'generated/summary.csv'
rows=list(csv.DictReader(p.open()));fields=list(rows[0]);fields=[('proposal_dlo' if f=='delta' else f) for f in fields]
for r in rows:
    arm=r['arm'];r['proposal_dlo']=summary[arm]['mean_dlo'] if arm in summary else '';r.pop('delta')
with p.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

for name in ['main','confidence_intervals']:
    p=ROOT/'tables'/f'{name}.tex';s=p.read_text(encoding='utf-8').replace('$\\Delta\\ell$','$\\Delta L$')
    s=s.replace('Completion on 100 strict-DFS sources', 'Teacher-forced proposal DLO and completion on 100 strict-DFS sources') if name=='main' and 'Teacher-forced proposal DLO' not in s else s
    if name=='confidence_intervals':
        s=s.replace('from 10,000 source-level bootstrap resamples.', 'from 100,000 source-level bootstrap resamples for DLO and 10,000 for F1.')
    p.write_text(s,encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.3,4.5),sharex=True)
for ax,mode,title in zip(axes,['natural','no_eos'],['EOS allowed','EOS suppressed']):
    for arm,label in names.items():
        r=summary[arm];v=float(r['mean_dlo']);lo=float(r['lower95']);hi=float(r['upper95'])
        length=stats[arm][mode+'_valid_length'];y=length['mean'];yl,yh=length['ci95']
        col='#21718b' if label.startswith('B') else '#c5772c'
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],yerr=[[y-yl],[yh-y]],fmt='o',ms=5,ecolor=col,color=col,alpha=.8,lw=.8,capsize=2,label=label)
    ax.set_title(title);ax.set_xlabel('Proposal DLO, teacher forced (natural log odds)');ax.set_ylabel('Mean valid prefix length (bricks)');ax.grid(alpha=.18);ax.set_xlim(8,68);ax.set_ylim(-5,105)
fig.suptitle('Reference sensitivity and independent generation',fontsize=12,fontweight='bold')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=6,frameon=False,fontsize=8)
fig.text(.5,.14,'DLO: 100 paired sources; valid length: 64 independent draws per checkpoint and stopping policy. Bars: 95% bootstrap intervals.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.19,1,.94))
for dest in [OUT,ROOT/'figures']:
    for ext in ['png','pdf']:fig.savefig(dest/f'proposal_dlo_vs_valid_length.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

# Regenerate the existing overview with corrected teacher-forced values.
from matplotlib import pyplot as plt
ids=list(names)
fig,ax=plt.subplots(figsize=(7.1,4.0));yy=list(range(len(ids)))[::-1]
for y,arm in zip(yy,ids):
    r=summary[arm];v=float(r['mean_dlo']);lo=float(r['lower95']);hi=float(r['upper95'])
    ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#c5762d',capsize=2,ms=4)
ax.set_yticks(yy,[names[a] for a in ids]);ax.set_xlabel('Proposal DLO, teacher forced');ax.grid(axis='x',alpha=.18);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'proposal_dlo_overview.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print('Updated 12 proposal DLO scores, report tables, and two plots.')
