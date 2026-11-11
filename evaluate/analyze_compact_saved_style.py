"""Brand-level paired analysis of the frozen compact study; no API calls."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import random
import statistics
import workflow_benchmark as w


def paired(differences, seed=20261021, samples=20000):
    n = len(differences)
    estimate = statistics.mean(differences)
    observed = abs(sum(differences))
    p = sum(abs(sum(v*s for v, s in zip(differences, signs))) >= observed-1e-12
            for signs in itertools.product((-1, 1), repeat=n)) / 2**n
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(differences, k=n)) for _ in range(samples))
    return {'mean_difference_percentage_points': 100*estimate,
            'bootstrap_95_percentile_ci_percentage_points': [100*boot[int(samples*.025)], 100*boot[int(samples*.975)]],
            'exact_two_sided_sign_flip_p': p, 'independent_brands': n,
            'bootstrap_draws': samples, 'seed': seed}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    corrected, last = [None]*len(values), 0
    for rank, index in enumerate(order):
        last = max(last, min(1, (len(values)-rank)*values[index]))
        corrected[index] = last
    return corrected


def analyze(out):
    protocol = w.load(out/'protocol.json')
    tasks = {t['task_id']: t for t in protocol['tasks']}
    expected = {(r['task_id'],r['condition'],r['repeat']) for r in protocol['schedule']}
    rows = [w.load(p) for p in (out/'scores').glob('*.json')]
    actual = {(r['task_id'],r['condition'],r['repeat']) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError(f'Complete analysis requires {len(expected)} unique scored requests; found {len(rows)}')
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        task = tasks[row['task_id']]
        if row['source_id'] != task['source_id']:
            raise ValueError('Source mapping mismatch')
        for name, value in [('task', row['task_metrics']['conservative_adherence']),
                            ('style', row['style_metrics']['style_retention_proxy']),
                            ('style_unknown', row['style_metrics']['unknown_rate'])]:
            grouped[row['condition']][row['source_id']][(row['task_id'], name)].append(value)
    brand_means = {}
    for condition, sources in grouped.items():
        brand_means[condition] = {}
        for source, groups in sources.items():
            brand_means[condition][source] = {metric: statistics.mean(
                statistics.mean(values) for (task_id, name), values in groups.items() if name == metric)
                for metric in ('task', 'style', 'style_unknown')}
    comparisons = []
    for other in ('saved_summary', 'task_only'):
        for metric in ('style', 'task'):
            brands = sorted(brand_means['saved_nine'])
            delta = [brand_means['saved_nine'][b][metric]-brand_means[other][b][metric] for b in brands]
            comparisons.append(dict(comparison='saved_nine-minus-'+other, metric=metric,
                **paired(delta), per_brand_difference={b:100*v for b,v in zip(brands,delta)}))
    corrections = holm([c['exact_two_sided_sign_flip_p'] for c in comparisons])
    for row, corrected in zip(comparisons, corrections):
        row['holm_p_four_primary_comparisons'] = corrected
    result = {'status':'complete_machine_proxy_analysis', 'images':len(rows),
        'independent_brands':len(protocol['sources']), 'human_evaluation':'not_collected_for_this_batch',
        'scope':protocol['interpretation'], 'brand_means':brand_means,
        'condition_means_percentage':{c:{m:100*statistics.mean(v[m] for v in sources.values())
            for m in ('task','style','style_unknown')} for c,sources in brand_means.items()},
        'primary_comparisons':comparisons}
    w.save(out/'analysis.json',result)
    print(json.dumps(result['condition_means_percentage'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    analyze(Path(args.out))
