"""Bounded, self-contained LDraw/MPD import into a connected BrickNet path."""
from __future__ import annotations
from pathlib import PurePosixPath


def ldraw_to_path(text: str, max_parts: int = 200):
    import numpy as np
    from bricknet import parse_ldr, parse_sample, sample_tree, serialize_tree
    from bricknet.data import load_catalog, load_aliases

    catalog=load_catalog();known=set(catalog.stem_to_id)|set(load_aliases())
    sections={};current=None;root_name=None;plain=[]
    for raw in text.splitlines():
        line=raw.strip();upper=line.upper()
        if upper.startswith('0 FILE '):
            current=line[7:].strip().replace('\\','/').lower()
            if not current or current in sections:raise ValueError('MPD 子模型名称为空或重复。')
            sections[current]=[]
            if root_name is None:root_name=current
        elif upper=='0 NOFILE':current=None
        elif current is not None:sections[current].append(line)
        elif root_name is None:plain.append(line)
    leaves=[]
    def expand(lines,transform,color,stack):
        if len(stack)>32:raise ValueError('MPD 子模型嵌套超过32层。')
        for line in lines:
            if not line.startswith('1 '):continue
            fields=line.split(maxsplit=14)
            if len(fields)!=15:raise ValueError('LDraw 零件行需要颜色、位置、旋转矩阵和文件名。')
            try:
                c=int(fields[1],0) if fields[1].lower().startswith('0x') else int(fields[1])
                values=np.asarray([float(x) for x in fields[2:14]])
            except ValueError as exc:raise ValueError('LDraw 数值字段无效。') from exc
            if not np.isfinite(values).all():raise ValueError('LDraw 不能含 NaN 或无穷坐标。')
            local=np.eye(4);local[:3,3]=values[:3];local[:3,:3]=values[3:].reshape(3,3)
            combined=transform@local;c=color if c==16 else c
            ref=fields[14].strip().replace('\\','/').lower()
            if ref.startswith('/') or '..' in PurePosixPath(ref).parts or ':' in ref:
                raise ValueError('LDraw 引用必须是自包含子模型或零件库名称。')
            stem=ref.removesuffix('.dat')
            if stem not in known and ref.startswith('parts/'):stem=stem[6:]
            if stem in known:
                nums=[*combined[:3,3],*combined[:3,:3].reshape(-1)]
                leaves.append(f'1 {c} '+ ' '.join(f'{x:.10g}' for x in nums)+' '+stem+'.dat')
                if len(leaves)>max_parts:raise ValueError(f'本界面最多导入{max_parts}个零件。')
            elif ref in sections:
                if ref in stack:raise ValueError('MPD 子模型存在循环引用。')
                expand(sections[ref],combined,c,stack+(ref,))
            else:raise ValueError(f'未找到零件或内嵌子模型：{ref}。请导出自包含MPD，或使用BrickNet零件库。')
    expand(sections[root_name] if root_name else plain,np.eye(4),7,(root_name,) if root_name else ())
    if not leaves:raise ValueError('文件中没有可导入的积木零件。')
    graph=parse_ldr('\n'.join(leaves)+'\n')
    if len(graph.components)!=1:
        raise ValueError(f'模型有{len(graph.components)}个未连接分量；请提供一个连通装配，不能静默丢弃其它部件。')
    try:path=serialize_tree(sample_tree(graph,method='bfs',seed=0))
    except (ValueError,KeyError) as exc:raise ValueError(f'模型含当前路径格式不支持的零件、颜色或连接：{exc}') from exc
    parsed=parse_sample(path)
    if parsed.error or len(parsed.tree.parts)!=len(leaves):raise ValueError('模型转换未能保留全部零件。')
    return path,{'imported_parts':len(leaves),'components':1,'root_normalized':True,'format':'LDraw/MPD'}
