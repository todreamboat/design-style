"""Reproduce main manuscript means from released final scores; no API calls."""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from decimal import Decimal, ROUND_HALF_UP

ARMS = ('task_only', 'saved_summary', 'saved_nine')
ROOT = Path(__file__).resolve().parents[1]

def balanced(rows, metric):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['source_id']][row['task_id']].append(float(row[metric]))
    return 100 * mean(mean(mean(v) for v in tasks.values()) for tasks in groups.values())

def main():
    with (ROOT/'reproduction/compact_study/final_scores.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 128
    keys = defaultdict(set)
    for row in rows:
        keys[(row['task_id'], row['repeat'])].add(row['condition'])
    complete = {k for k,v in keys.items() if v == set(ARMS)}
    paired = [r for r in rows if (r['task_id'],r['repeat']) in complete]
    assert len(complete) == 24 and len(paired) == 72
    expected = {'all':[40.63,75.50,83.49], 'paired':[43.68,73.32,83.88]}
    print('scope,condition,images,task_percent,style_percent')
    for scope, selected in [('all',rows),('paired',paired)]:
        for index, arm in enumerate(ARMS):
            subset = [r for r in selected if r['condition'] == arm]
            style = balanced(subset,'style_score')
            assert abs(style-expected[scope][index]) < .0051
            display = Decimal(str(round(style, 10))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
            print(f'{scope},{arm},{len(subset)},{balanced(subset,"task_score"):.2f},{display}')

if __name__ == '__main__': main()
