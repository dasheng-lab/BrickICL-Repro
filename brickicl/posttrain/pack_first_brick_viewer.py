"""Stage every unified completion for a case/prime/condition/sample board."""
import json
from pathlib import Path
from bricknet import score_text
from brickicl.posttrain.paths import model_label
from brickicl.viz import pack_probe as pp
from brickicl.viz.lazy_catalog import _write_json
import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('runs/history_first_brick06'))
    args = parser.parse_args(argv)
    ROOT=args.run_dir;assert (ROOT/'COMPLETED.json').exists()
    NAMES={15:'桌子／拱门',48:'电线杆',57:'城市建筑',58:'灰色建筑',85:'交通路口',95:'多级火箭',105:'双人控制台',113:'机械臂装置',114:'狐狸',119:'砖墙绿植',175:'办公室前台',183:'灰色柱子',252:'粉衣人物',258:'蘑菇花盆',290:'双体潜航器',433:'AT-ST步行机',458:'出租车',475:'未来载具'}
    ARMS={'frozen_pt':'官方 PT','post_pt':'M-only PT · step 1750','fullg_pt':'Full-G PT · step 500',
          'frozen_sft':'官方 SFT','post_sft':'M-only SFT · step 2188','fullg_sft':'Full-G SFT · step 250'}
    ARMS={a:model_label(a,json.loads((ROOT/f'provenance.{a}.json').read_text())['adapters']) for a in ARMS}
    CONDITIONS={'original':'正常','stretched':'加长','ablated':'删除参照'}
    def read(p):return [json.loads(s) for s in p.open()]
    inputs={p:{r['id']:r for r in read(ROOT/f'p{p}_triads.jsonl')} for p in (0,1)}
    results={p:{a:{r['id']:r for r in read(ROOT/f'p{p}.{a}.jsonl')} for a in ARMS} for p in (0,1)}
    sources=json.loads((ROOT/'sources.json').read_text());sources.sort(key=lambda r:([114,458].index(r['source']) if r['source'] in [114,458] else 2+r['source']))
    board={'schema':1,'arms':ARMS,'samples':4,'source_count':18,'sources':{},'source_order':[str(s['source']) for s in sources]}
    packer=pp.Packer()
    for source in sources:
        sid=source['source'];case={'label':f'#{sid} · {NAMES[sid]}','caption':source['caption'],'primes':{}}
        for prime in (0,1):
            prime_label='给第一块' if prime else '不给第一块';task='completion_given_head' if prime else 'completion_no_head'
            group=f'{prime_label} · #{sid} {NAMES[sid]}'
            conditions={}
            for cond,label in CONDITIONS.items():
                r=inputs[prime][f'val:{sid}:{cond}'];target=r['target_'+r['expected']]
                info={'label':label,'prefix_parts':r['n_prefix'],'target_parts':r['k_'+r['expected']],
                      'input_sha256':r['prefix_sha256'],'outputs':{}}
                stem=f'unified/p{prime}/{sid}/{cond}'
                for kind,text in [('input',r['prefix']),('reference',r['prefix']+target)]:
                    rel=f'{stem}_{kind}.mpd'
                    packer.add(rel,text,r['n_prefix'],group=group,title=f'{label} · '+('共同输入' if kind=='input' else '参考答案'),
                        caption=source['caption'],is_reference=True,collisions=score_text(text)[2],
                        note=f'{prime_label}；六个模型接收相同装配输入，不含文字caption或额外示范。案例说明仅用于展示。输入{r["n_prefix"]}件，待补{r["k_"+r["expected"]]}件。')
                    packer.entries[-1].update(task=task,evaluation_run='history_first_brick06',source_key=r['source_key'],prime=prime,sample_index=None)
                    info[kind]=rel
                for arm,title in ARMS.items():
                    out=results[prime][arm][r['id']];assert len(out['samples'])==4
                    assert out['seed']==results[prime]['frozen_pt'][r['id']]['seed']
                    assert out['max_new_tokens']==results[prime]['frozen_pt'][r['id']]['max_new_tokens']
                    entries=[]
                    for i,s in enumerate(out['samples']):
                        rel=f'{stem}_{arm}_s{i+1}.mpd'
                        note=(f'{prime_label}；实际采样{i+1}/4，未筛选。正常/加长/删除参照均使用固定的同一来源；两种首块设置共享随机种子与token上限。'
                              f' 输入{r["n_prefix"]}件，生成{s["n_generated"]}件，目标{r["k_"+r["expected"]]}件；'
                              f'位姿F1={s["pose_f1"]:.3f}，零件F1={s["node_f1"]:.3f}；整段解析={s["parse_ok"]}；EOS={s["eos_emitted"]}。')
                        packer.add(rel,r['prefix']+s['text'],r['n_prefix'],group=group,title=f'{label} · {title} · 样本{i+1}',
                            caption=source['caption'],note=note,collisions=s['collisions'])
                        packer.entries[-1].update(task=task,evaluation_run='history_first_brick06',source_key=r['source_key'],prime=prime,
                            arm=arm,sample_index=i,evaluation_metrics={k:s[k] for k in ['pose_f1','node_f1','exact','parse_ok','eos_emitted']})
                        entries.append({'file':rel,**{k:s[k] for k in ['pose_f1','node_f1','parse_ok','n_generated','exact','eos_emitted']}})
                    info['outputs'][arm]=entries
                conditions[cond]=info
            case['primes'][str(prime)]={'prime':prime,'conditions':conditions}
        board['sources'][str(sid)]=case
        print('PACKED',sid,NAMES[sid],len(packer.entries),flush=True)
    assert len(packer.entries)==2808
    _write_json(ROOT/'viewer_entries.json',packer.entries)
    _write_json(ROOT/'unified_compare.json',board)
    (ROOT/'VIEWER_STAGED.json').write_text(json.dumps({'views':2808,'sources':18,'samples_per_case_condition_model':4},indent=2)+'\n')

if __name__ == '__main__':
    main()
