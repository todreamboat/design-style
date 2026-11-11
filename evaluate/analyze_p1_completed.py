"""Import author-confirmed labels and analyze the completed matched format ablation.

No API calls. Original human file and machine scores are never overwritten.
"""
import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import p1_supplement as p
import workflow_benchmark as w
from analyze_completed_comparison import inference, METRICS


def no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Duplicate JSON key: '+key)
        obj[key] = value
    return obj


def agreement(rows, categories, weighted=False):
    if not rows:
        return dict(judgments=0, exact_agreement=None, kappa=None)
    table = np.zeros((len(categories),len(categories)),dtype=int)
    for r in rows:
        table[categories.index(r['human']),categories.index(r['machine'])] += 1
    n=int(table.sum())
    probs=table/n
    expected=np.outer(probs.sum(axis=1),probs.sum(axis=0))
    weights=(1-np.abs(np.arange(len(categories))[:,None]-np.arange(len(categories))[None,:])/(len(categories)-1)
             if weighted else np.eye(len(categories)))
    observed=float((probs*weights).sum())
    chance=float((expected*weights).sum())
    kappa=(observed-chance)/(1-chance) if 1-chance>1e-12 else None
    return dict(judgments=n, matches=int(np.trace(table)), exact_agreement=float(np.trace(table)/n),
        kappa=kappa, kappa_type='linear_weighted' if weighted else 'unweighted',
        category_order=categories, confusion_human_rows_machine_columns=table.tolist(),
        human_distribution=dict(Counter(r['human'] for r in rows)),
        machine_distribution=dict(Counter(r['machine'] for r in rows)))


def audit_human(path):
    raw=p.json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=no_duplicates)
    mapping=w.load(p.OUT/'blind_mapping_DO_NOT_OPEN_BEFORE_RATING.json')['cases']
    if raw['protocol']!='p1-blind-48-v1' or set(raw['cases'])!={m['case_id'] for m in mapping}:
        raise ValueError('Expected exactly the frozen 48 cases')
    pairs=[]
    case_scores=[]
    for item in mapping:
        labels=raw['cases'][item['case_id']]
        fields=['t1','t2','p1','p2','q1','q2']+item['eligible_dimensions']
        if set(labels)!=set(fields+['evidence']):
            raise ValueError('Missing/extra fields in '+item['case_id'])
        record=w.load(p.ROOT/'seed'/(item['run_id']+'.json'))
        if w.digest(record['output']['path'])!=item['output_sha256']:
            raise ValueError('Blind-reviewed image changed')
        machine=w.load(p.ROOT/'seed'/(item['run_id']+'.scores.json'))
        machine_labels={r['constraint_id']:r['status'] for r in machine['raw_constraints']['checks']}
        machine_labels.update({r['dimension']:r['status'] for r in machine['raw_style']['style_checks']})
        normalized={}
        for field in fields:
            group={'t':'task','p':'presentation','q':'artifact'}.get(field[0]) if len(field)==2 else 'style'
            human=labels[field]
            if human=='':
                human='retained' if group=='style' else 'met'
            allowed=['retained','partial','not_retained','unknown'] if group=='style' else ['met','violated','unknown']
            if human not in allowed:
                raise ValueError('Invalid label in '+item['case_id']+' '+field)
            normalized[field]=human
            pairs.append(dict(case_id=item['case_id'],run_id=item['run_id'],source_id=machine['source_id'],
                task_id=item['task_id'],condition=item['condition'],group=group,field=field,
                human=human,machine=machine_labels[field],agrees=human==machine_labels[field],
                human_evidence=labels['evidence']))
        scores={g:sum(normalized[f]== 'met' for f in fields if f.startswith(prefix) and len(f)==2)/2
                for g,prefix in [('task','t'),('presentation','p'),('artifact','q')]}
        scores['style']=np.mean([{'retained':1,'partial':.5,'not_retained':0,'unknown':0}[normalized[f]]
                                for f in item['eligible_dimensions']])
        case_scores.append(dict(source=machine['source_id'],condition=item['condition'],task_id=item['task_id'],
            **{m:float(v) for m,v in scores.items()}))
    results={g:agreement([r for r in pairs if r['group']==g],['met','violated','unknown'])
             for g in ['task','presentation','artifact']}
    results['style_all_labels']=agreement([r for r in pairs if r['group']=='style'],
        ['not_retained','partial','retained','unknown'])
    results['style_known_ordinal']=agreement([r for r in pairs if r['group']=='style' and 'unknown' not in (r['human'],r['machine'])],
        ['not_retained','partial','retained'],weighted=True)
    source_order=sorted({r['source_id'] for r in pairs})
    boot=np.asarray(list(itertools.product(range(6),repeat=6)))
    for group,result_key in [('task','task'),('presentation','presentation'),('artifact','artifact'),('style','style_all_labels')]:
        rates=[np.mean([r['agrees'] for r in pairs if r['group']==group and r['source_id']==s]) for s in source_order]
        results[result_key]['source_bootstrap_agreement_ci95']=np.quantile(np.asarray(rates)[boot].mean(axis=1),[.025,.975]).tolist()
    means={c:{m:float(np.mean([r[m] for r in case_scores if r['condition']==c]))
              for m in ['task','style','presentation','artifact']} for c in p.SELECTED}
    machine_means={c:{m:float(np.mean([w.load(p.ROOT/'seed'/(item['run_id']+'.scores.json'))[METRICS[m][0]][METRICS[m][1]]
        for item in mapping if item['condition']==c])) for m in ['task','style','presentation','artifact']} for c in p.SELECTED}
    payload=dict(status='author_single_reviewer_audit_completed',label_file=w.file_record(path),
        mapping_sha256=w.digest(p.OUT/'blind_mapping_DO_NOT_OPEN_BEFORE_RATING.json'),
        cases=48,source_blocks=6,reviewers=1,reviewer_qualification='not independently verified',
        blank_policy='Author confirmed all unchecked entries were reviewed and mean met/retained; explicit labels unchanged',
        default_provenance='Export does not preserve original empty-versus-clicked distinction; counts cannot be reconstructed',
        evidence_provided=sum(bool(raw['cases'][m['case_id']]['evidence'].strip()) for m in mapping),
        blinded_interface='Condition names and machine scores hidden in supplied interface; external knowledge not certified',
        kind='single-author attribute audit, not designer satisfaction study or independent ground truth',
        agreement=results,human_means=means,machine_means_same48=machine_means,case_scores=case_scores,
        disagreements=[r for r in pairs if not r['agrees']],
        qualification='High agreement alone does not establish reliability when one category dominates; retain kappa and distributions')
    w.save(p.OUT/'human_audit_analysis.json',payload)
    p.write_csv(p.OUT/'human_machine_pairs.csv',pairs)
    if payload['disagreements']:
        p.write_csv(p.OUT/'human_machine_disagreements.csv',payload['disagreements'])
    return payload


def analyze_flat():
    out=p.OUT/'flat_base'
    if w.load(out/'status.json')['status']!='completed':
        raise ValueError('Ablation incomplete')
    protocol=w.load(out/'frozen_protocol.json')
    plans={(r['task_id'],r['repeat']):r for r in protocol['planned_runs']}
    rows=[w.load(f) for f in sorted(out.glob('*.scores.json'))]
    if len(rows)!=24 or {(r['task_id'],r['repeat']) for r in rows}!=set(plans):
        raise ValueError('Missing/duplicate ablation scores')
    groups=defaultdict(list)
    for row in rows:
        plan=plans[row['task_id'],row['repeat']]
        record=w.load(out/(row['run_id']+'.json'))
        assert row['condition']=='D_base' and record['image_seed']==plan['image_seed']
        if w.digest(record['output']['path'])!=record['output']['sha256'] or record['prompt']!=plan['prompt']:
            raise ValueError('Ablation evidence hash mismatch')
        groups[row['source_id'],row['task_id']].append(row)
    task_means=[]
    for (s,t),members in groups.items():
        assert len(members)==2
        task_means.append(dict(source=s,task_id=t,**{m:float(np.mean([r[a][b] for r in members])) for m,(a,b) in METRICS.items()}))
    sources=sorted({r['source'] for r in task_means})
    source_means=[dict(source=s,condition='D_base',**{m:float(np.mean([r[m] for r in task_means if r['source']==s])) for m in METRICS}) for s in sources]
    original=w.load(p.OUT/'analysis.json')
    base={r['source']:r for r in original['source_means'] if r['condition']=='C_no_priority'}
    contrasts={m:inference([base[r['source']][m]-r[m] for r in source_means]) for m in ['task','style']}
    previous=0.
    ordered=sorted(contrasts,key=lambda m:contrasts[m]['exact_sign_flip_p'])
    for i,m in enumerate(ordered):
        previous=max(previous,min(1.,(2-i)*contrasts[m]['exact_sign_flip_p']))
        contrasts[m]['holm_p_2_format_endpoints']=previous
    budget=w.load(out/'budget_ledger.json')
    payload=dict(status='completed',outputs=24,source_blocks=6,
        contrast='C_no_priority minus D_base; unedited identical field values, shared wrapper and seeds',
        summary={m:float(np.mean([r[m] for r in source_means])) for m in METRICS},
        source_means=source_means,contrasts=contrasts,
        by_task_kind={k:{m:float(np.mean([r[m] for r in task_means if r['task_id'].endswith(k)])) for m in METRICS}
            for k in ['appearance','structure']},
        returned_cost_usd=sum(r.get('returned_cost_usd',0) for r in budget['requests']),
        accounted_or_reserved_usd=sum(r['charged_or_reserved_usd'] for r in budget['requests']),
        policy='supplementary exploratory two-sided six-source sign-flip; Holm across the two format endpoints; percentile source bootstrap; no independent human audit of this arm')
    w.save(p.OUT/'flat_ablation_analysis.json',payload)
    p.write_csv(p.OUT/'flat_source_means.csv',source_means)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path,required=True)
    args=parser.parse_args()
    human=audit_human(args.labels.resolve())
    flat=analyze_flat()
    print(p.json.dumps(dict(human_agreement=human['agreement'],human_means=human['human_means'],
        disagreements=len(human['disagreements']),flat=flat),ensure_ascii=False,indent=2))
