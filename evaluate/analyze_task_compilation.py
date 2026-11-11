"""Offline verification and source-block descriptive summary of the paired test."""
from collections import defaultdict
from statistics import mean
import json
import run_task_compilation as experiment

w=experiment.w
OUT=experiment.OUT


def analyze():
    if w.load(OUT/'status.json')['status']!='completed':
        raise ValueError('Experiment incomplete; no final summary')
    protocol=w.load(OUT/'protocol.json')
    plans=w.load(OUT/'generation_plan.json')
    assert len(plans)==48
    for record in protocol['sources'].values():
        assert w.digest(record['path'])==record['sha256']
    for task in protocol['tasks']:
        saved=w.load(protocol['sources'][task['source_id']]['path'])['style_summary']['designer_style_summary']
        experiment.validate_plan(w.load(OUT/(task['task_id']+'.task_plan.json')),saved,task)
    groups=defaultdict(list)
    paired=defaultdict(dict)
    for plan in plans:
        record=w.load(OUT/(plan['run_id']+'.json'))
        assert record['status']=='completed' and record['input_records']==[]
        assert w.digest(record['output']['path'])==record['output']['sha256']
        assert w.text_hash(record['prompt'])==plan['prompt_sha256']
        score=w.load(OUT/(plan['run_id']+'.scores.json'))
        assert (score['run_id'],score['condition'],score['repeat'])==(plan['run_id'],plan['condition'],plan['repeat'])
        assert len(score['raw_constraints']['checks'])==6
        paired[plan['task_id'],plan['repeat']][plan['condition']]=plan['seed']
        groups[plan['source_id'],plan['condition'],plan['task_id']].append(score)
    for pair in paired.values():
        assert set(pair)=={'direct_nine','compiled_nine'} and len(set(pair.values()))==1
    metrics={'task':('task_metrics','conservative_adherence'),
             'style':('style_metrics','style_retention_proxy'),
             'unknown':('style_metrics','unknown_rate')}
    sources=defaultdict(list)
    for (source,condition,task),rows in groups.items():
        assert len(rows)==2
        sources[source,condition].append({m:mean(r[a][b] for r in rows) for m,(a,b) in metrics.items()})
    by_source=[]
    for (source,condition),values in sorted(sources.items()):
        assert len(values)==2
        by_source.append(dict(source_id=source,condition=condition,
            **{m:100*mean(r[m] for r in values) for m in metrics}))
    summary={c:{m:mean(r[m] for r in by_source if r['condition']==c) for m in metrics}
             for c in ('direct_nine','compiled_nine')}
    ledger=w.load(OUT/'budget_ledger.json')
    result=dict(status='completed_verified',images=48,paired_task_repeats=len(paired),independent_sources=6,
        summary_percent=summary,source_means_percent=by_source,
        differences_pp={m:summary['compiled_nine'][m]-summary['direct_nine'][m] for m in metrics},
        returned_cost_usd=sum(r.get('returned_cost_usd',0) for r in ledger['requests']),
        accounted_usd=sum(r['charged_or_reserved_usd'] for r in ledger['requests']),
        interpretation='Development comparison of complete prompt compilation workflow; not held-out confirmation, no causal isolation of individual planning rules; unchanged machine proxy with known limitations')
    w.save(OUT/'comparison_analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': analyze()
