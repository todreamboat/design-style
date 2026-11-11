"""Post-hoc exploratory paired source analysis; never treats 240 images as independent n."""
import itertools
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import workflow_benchmark as w

ROOT = w.HERE/'results/nine_comparison_run_20261006'
METRICS = {'task': ('task_metrics', 'conservative_adherence'),
           'style': ('style_metrics', 'style_retention_proxy'),
           'presentation': ('presentation_metrics', 'conservative_adherence'),
           'artifact': ('artifact_diagnostics', 'conservative_adherence'),
           'task_unknown': ('task_metrics', 'unknown_rate'),
           'style_unknown': ('style_metrics', 'unknown_rate')}


def inference(differences):
    d = np.array(differences)
    boot = np.array(list(itertools.product(range(len(d)), repeat=len(d))))
    ci = np.quantile(d[boot].mean(axis=1), [.025, .975])
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    p = float(np.mean(np.abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-12))
    return dict(delta=float(d.mean()), bootstrap_ci95=ci.tolist(),
                exact_sign_flip_p=p, positive_sources=int(sum(d>1e-12)),
                negative_sources=int(sum(d < -1e-12)), source_differences=d.tolist())


def analyze(stage):
    out = ROOT/stage
    schedule = {f'{t}_{c}_{r}' for t,c,r in w.load(out/'schedule.json')}
    rows = [w.load(p) for p in sorted(out.glob('*.scores.json'))]
    assert {r['run_id'] for r in rows} == schedule and len(rows) == len(schedule)
    assert w.load(out/'status.json')['status'] == 'completed'
    for row in rows:
        record=w.load(out/(row['run_id']+'.json'))
        assert w.digest(record['output']['path']) == record['output']['sha256']
    groups=defaultdict(list)
    for row in rows:
        groups[row['source_id'],row['task_id'],row['condition']].append(row)
    task_rows=[]
    for (source,task,c),group in sorted(groups.items()):
        assert len(group)==2
        task_rows.append(dict(source=source,task_id=task,condition=c,
            **{m:float(np.mean([r[a][b] for r in group])) for m,(a,b) in METRICS.items()}))
    source_groups=defaultdict(list)
    for row in task_rows:
        source_groups[row['source'],row['condition']].append(row)
    source_rows={key:{m:float(np.mean([r[m] for r in group])) for m in METRICS}
                 for key,group in source_groups.items()}
    sources=sorted({s for s,c in source_rows})
    conditions=sorted({c for s,c in source_rows})
    assert len(sources)==6
    summary={c:{m:float(np.mean([source_rows[s,c][m] for s in sources])) for m in METRICS} for c in conditions}
    by_kind={kind:{c:{m:float(np.mean([r[m] for r in task_rows if r['condition']==c and r['task_id'].endswith(kind)]))
                            for m in METRICS} for c in conditions} for kind in ('appearance','structure')}
    contrasts={c:{m:inference([source_rows[s,'C'][m]-source_rows[s,c][m] for s in sources])
                       for m in ('task','style')} for c in conditions if c!='C'}
    return dict(outputs=len(rows),source_blocks=6,summary=summary,by_task_kind=by_kind,
                source_means=[dict(source=s,condition=c,**v) for (s,c),v in source_rows.items()],contrasts=contrasts)


def main():
    results={stage:analyze(stage) for stage in ('seed','flux')}
    # One exploratory comparison family, across both backends and both endpoints.
    tests=[v for stage in results.values() for contrast in stage['contrasts'].values() for v in contrast.values()]
    order=sorted(range(len(tests)),key=lambda i:tests[i]['exact_sign_flip_p'])
    previous=0
    for rank,index in enumerate(order):
        previous=max(previous,min(1,(len(tests)-rank)*tests[index]['exact_sign_flip_p']))
        tests[index]['holm_p_all_16_tests']=previous
    results['analysis_policy']=dict(independent_unit='six matched reference-source/brand blocks, convenience development sample',
        repeated_outputs='two generations averaged within task, then two tasks within source',
        inference='post-hoc exploratory two-sided exact source sign-flip (symmetry/exchangeability assumption)',
        uncertainty='percentile source bootstrap, exact enumeration of 6^6 resamples; small-n descriptive interval',
        multiplicity='Holm across all 16 task/style contrasts including both backends',
        human_calibration=False,formal_test=False,numpy_version=np.__version__,
        limitations='Sources are not randomly sampled; not cross-series generalization or proof of nine-dimension necessity')
    w.save(ROOT/'comparison_analysis.json',results)
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
