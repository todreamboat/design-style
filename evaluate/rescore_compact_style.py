"""Re-evaluate existing images only, with explicit image roles and shared budget."""
import argparse
from pathlib import Path
import workflow_benchmark as w
from run_compact_saved_style import Budget
from compact_retry import AutoBudget, validated_cache


def corrected_template(template):
    old = '图1和图2为服装风格参考，图3为生成结果。'
    new = '图1至图4为服装风格参考，图5为生成结果。'
    if not template.startswith(old):
        raise ValueError('Unexpected original image-role prefix')
    result = new + template[len(old):]
    assert result.startswith(new) and old not in result
    assert result[len(new):] == template[len(old):]
    return result


class ContinuationBudget(Budget):
    """Separate incremental authorization; prior spending remains untouched."""
    def __init__(self, out):
        self.out, self.phase_cap = out, 10
        self.path = out/'budget_ledger.json'
        if self.path.exists():
            self.data = w.load(self.path)
            if self.data['cap_usd'] != 10:
                raise ValueError('Continuation ceiling changed')
        else:
            self.data = dict(cap_usd=10, prior_probe_usd=0, requests=[],
                authorization='User approved continued existing-image style evaluation, incremental ceiling 10 USD',
                excludes='All prior spending; no image generation')
            w.save(self.path, self.data)


def main(out, execute, continuation=False):
    protocol = w.load(out/'protocol.json')
    template = corrected_template(protocol['prompts']['style_judge'])
    cfg = dict(protocol['cfg'], vlm_model=protocol['cfg']['judge_model'], vlm_extra={})
    tasks = {t['task_id']: t for t in protocol['tasks']}
    sources = {s['source_id']: s for s in protocol['sources']}
    jobs = []
    for record in sorted((out/'outputs').glob('*.json')):
        row = w.load(record)
        task = tasks[row['task_id']]
        refs = [w.ROOT.parent/'data'/f['path'] for f in sources[task['source_id']]['history']]
        image = Path(row['output']['path'])
        assert len(refs) == 4 and w.digest(image) == row['output']['sha256']
        paths = refs + [image]
        prompt = template.format(excluded_dimensions=w.json.dumps(task['style_excluded_dimensions']),
                                 eligible_dimensions=w.json.dumps(task['eligible_style_dimensions']))
        assert prompt.startswith('图1至图4为服装风格参考，图5为生成结果。')
        jobs.append((record.stem, row, task, paths, prompt))
    assert len(jobs) == 128
    target = out/'style_rescore_v2'
    amendment = dict(original_protocol_sha256=w.digest(out/'protocol.json'),
                     reason='Correct image roles only; preserve metric, model, images and exclusions',
                     template=template, images=len(jobs), generation_calls=0,
                     image_order='four historical references, then generated image at position 5',
                     budget='shared original 25 USD ceiling; old scores unchanged')
    if (target/'protocol.json').exists():
        assert w.load(target/'protocol.json') == amendment
    else:
        w.save(target/'protocol.json', amendment)
    base = ContinuationBudget(target/'continuation_10usd') if continuation else Budget(out, 25)
    budget = AutoBudget(base)
    budget.authorized_network_permission_recovery = execute
    print(f'Validated {len(jobs)} fixed image mappings; accounted ${budget.spent():.4f}/{base.phase_cap}', flush=True)
    if not execute:
        return
    w.load_local_credentials()
    try:
        for name, row, task, paths, prompt in jobs:
            def validate(raw):
                value = w.strict_json(raw) if isinstance(raw, str) else raw
                w.style_checks(value, task['eligible_style_dimensions'])
                return value
            value = validated_cache(target/'judgments'/f'{name}.json', budget, 'text', cfg,
                                    prompt, paths, validate)
            w.save(target/'scores'/f'{name}.json', dict(run_id=name,
                task_id=row['task_id'], condition=row['condition'], repeat=row['repeat'],
                source_id=task['source_id'], image_mapping=[w.file_record(p) for p in paths],
                style_metrics=w.style_checks(value, task['eligible_style_dimensions'])))
            print('Rescored', name, f'accounted ${budget.spent():.4f}', flush=True)
        w.save(target/'status.json', dict(state='complete', completed=len(jobs), accounted_usd=budget.spent()))
    except Exception as exc:
        completed = len(list((target/'scores').glob('*.json')))
        w.save(target/'status.json', dict(state='stopped', completed=completed,
            error_type=type(exc).__name__, error=str(exc)[:250], accounted_usd=budget.spent()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute-api', action='store_true')
    parser.add_argument('--continuation-10usd', action='store_true')
    args = parser.parse_args()
    main(args.out, args.execute_api, args.continuation_10usd)
