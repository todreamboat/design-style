"""Offline descriptive comparison; historical batches, no inferential claims."""
from collections import defaultdict
import json
from statistics import mean
import run_retained_style as study

w=study.w
out=study.OUT


def analyze():
    if w.load(out/'status.json')['status']!='completed':
        raise ValueError('Do not summarize incomplete experiments')
    rows=[]
    plans=w.load(out/'generation_plan.json')
    assert len(plans)==24
    for plan in plans:
        record=w.load(out/(plan['run_id']+'.json'))
        assert record['status']=='completed' and record['input_records']==[]
        assert w.text_hash(record['prompt'])==plan['prompt_sha256']
        assert w.digest(record['output']['path'])==record['output']['sha256']
        assert w.digest(plan['saved_summary']['path'])==plan['saved_summary']['sha256']
        old=w.load(study.s.OUT/(plan['paired_existing_run']+'.json'))
        assert old['seed']==record['seed'] and old['task_id']==record['task_id']
        score=w.load(out/(plan['run_id']+'.scores.json'))
        assert (score['run_id'],score['condition'])==(record['run_id'],record['condition'])
        rows.append(score)
    for condition in ('saved_nine','saved_summary','task_only'):
        old=[w.load(p) for p in study.s.OUT.glob('*.scores.json')]
        rows.extend(r for r in old if r['condition']==condition)
    metrics={'task':('task_metrics','conservative_adherence'),
             'style':('style_metrics','style_retention_proxy'),
             'style_unknown':('style_metrics','unknown_rate')}
    groups=defaultdict(list)
    for r in rows:
        groups[r['condition'],r['source_id'],r['task_id']].append(r)
    source_values=defaultdict(list)
    for (condition,source,task),values in groups.items():
        assert len(values)==2
        source_values[condition,source].append({m:mean(v[a][b] for v in values)
                                             for m,(a,b) in metrics.items()})
    by_source=[]
    for (condition,source),values in sorted(source_values.items()):
        assert len(values)==2
        by_source.append(dict(condition=condition,source_id=source,
            **{m:mean(v[m] for v in values) for m in metrics}))
    summary={}
    for c in sorted({r['condition'] for r in rows}):
        src=[r for r in by_source if r['condition']==c]
        assert len(src)==6
        summary[c]={m:100*mean(v[m] for v in src) for m in metrics}
    budget=w.load(out/'budget_ledger.json')
    result=dict(status='completed_verified',images=24,source_units=6,summary_percent=summary,
        source_means=by_source,returned_cost_usd=sum(r.get('returned_cost_usd',0) for r in budget['requests']),
        accounted_usd=sum(r['charged_or_reserved_usd'] for r in budget['requests']),
        interpretation='Descriptive development rerun; historical comparator batch, not causal isolation or held-out confirmation')
    w.save(out/'comparison_analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    analyze()
