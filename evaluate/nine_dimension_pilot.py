"""Bounded nine-dimension development pilot; legacy eight-field results are not reused."""
import argparse
import csv
import io
import json
from pathlib import Path
import random
from PIL import Image
import workflow_benchmark as w

CONDITIONS = {
    'A': 'direct_reference_images',
    'B': 'free_summary',
    'B_reconciled': 'task_reconciled_free_summary',
    'C': 'original_nine_fields_with_override',
    'D': 'same_resolved_values_without_field_labels',
    'C_no_priority': 'nine_fields_without_override',
}
RECONCILE = ('以下是服装参考图的普通风格摘要：{summary}\n'
             '当前预先指定的修改要求：{explicit}\n'
             '只修改摘要中与指定要求直接冲突的表述，使其符合要求；保留其他可见属性、差异及未知表述。'
             '不添加新的设计属性，不重新组织成固定字段，不猜测隐藏结构、品牌或材质成分。'
             '只输出修订后的普通中文段落，不输出JSON或分析过程。总字符不超过480。')


def render(task, condition, profile, summary, reconciled, prompts):
    if condition == 'B_reconciled':
        return w.render(task, 'B', profile, reconciled, prompts)
    if condition in CONDITIONS:
        return w.render(task, condition, profile, summary, prompts)
    raise ValueError('Unknown nine-dimension condition')


class Budget:
    def __init__(self, out, cap):
        if not 0 < cap <= 5:
            raise ValueError('Initial probe budget must be in (0,5] USD')
        self.out, self.cap, self.rows = out, cap, []

    def call(self, kind, cfg, prompt, paths, seed=None):
        # Conservative reservations, not a provider-enforced billing guarantee.
        reserve = .05 if kind == 'image' else .25
        if sum(r['charged_or_reserved_usd'] for r in self.rows) + reserve > self.cap:
            raise ValueError('Budget reservation cap reached before dispatch')
        call_out = self.out / 'calls' / f'request_{len(self.rows)+1:03d}'
        row = {'kind': kind, 'reservation_usd': reserve, 'charged_or_reserved_usd': reserve,
               'status': 'dispatched', 'model': cfg['image_model'] if kind == 'image' else cfg['vlm_model']}
        self.rows.append(row)
        w.save(self.out / 'budget_ledger.json', {'cap_usd': self.cap, 'requests': self.rows})
        client = w.Client(dict(cfg, min_interval_seconds=4), call_out)
        try:
            result = client.image(prompt, paths, seed) if kind == 'image' else client.text(prompt, paths)
            row['status'] = 'returned'
            return result
        finally:
            for path in call_out.glob('*.metadata.json'):
                usage = w.load(path).get('usage')
                if isinstance(usage, dict) and isinstance(usage.get('cost'), (int, float)):
                    row['returned_cost_usd'] = usage['cost']
                    row['charged_or_reserved_usd'] = usage['cost']
            w.save(self.out / 'budget_ledger.json', {'cap_usd': self.cap, 'requests': self.rows,
                'cost_note': 'Failed requests without returned usage retain reservations; real billing may differ'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--candidates')
    p.add_argument('--kit')
    p.add_argument('--out', required=True)
    p.add_argument('--execute-api', action='store_true')
    p.add_argument('--budget-usd', type=float, default=5)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.mode == 'prepare':
        candidates = w.load(a.candidates)
        candidate = candidates[1]  # Frozen deterministic pilot, never chosen for method scores.
        task = {'task_id': 'nine_dev_b2_color', 'split': 'development', 'brand': candidate['brand'],
                'brief': '借鉴两张参考图设计一件主体为蓝色的外套。',
                'hard_constraints': ['外套主体为蓝色', '正面展示完整外套'],
                'preserve': [], 'explicit_fields': {'color_tone_usage_characteristics': '外套主体蓝色；其他服装配色依据参考'},
                'style_excluded_dimensions': ['overall_style_characteristics'],
                'reference_images': candidate['reference_images'], 'evaluation_images': candidate['evaluation_images'],
                'task_origin': 'Researcher-authored exploratory brief; not enterprise or standard benchmark task',
                'eligibility_status': 'pending_visual_review; no human ground truth',
                'primary_question': 'Does explicit field-based conflict resolution help relative to prose reconciliation?'}
        cfg, prompts = w.load(w.HERE / 'workflow_config.json'), w.load(w.HERE / 'workflow_prompts.json')
        cfg.update(protocol_version='original-nine-development-v1-20261003', repeats=1,
                   conditions=list(CONDITIONS), image_seed_base=20261003, max_live_outputs=6)
        w.validate_tasks([task], cfg)
        w.save(out / 'config.json', cfg)
        w.save(out / 'tasks.json', {'tasks': [task], 'status': 'exploratory_probe_not_formal_test'})
        w.save(out / 'prompts.json', dict(prompts, study_reconciliation=RECONCILE))
        w.save(out / 'code_snapshot.json', {name: w.file_record(w.HERE / name) for name in ['nine_dimension_pilot.py', 'workflow_benchmark.py']})
        w.save(out / 'condition_registry.json', CONDITIONS)
        planned_tasks = []
        for i, candidate_set in enumerate(candidates):
            for kind, brief, explicit in (
                ('color', '借鉴参考图设计一件蓝色主体的外套。', {'color_tone_usage_characteristics': '外套主体蓝色，其他配色参考原图'}),
                ('waist', '借鉴参考图设计一件腰部明确收束的外套。', {'waistline_treatment_characteristics': '腰部明确收束'}),
            ):
                planned_tasks.append(dict(task, task_id=f'nine_candidate_{i+1}_{kind}',
                    brand=candidate_set['brand'], brief=brief, explicit_fields=explicit,
                    hard_constraints=['外套主体为蓝色' if kind == 'color' else '外套腰部明确收束', '正面展示完整外套'],
                    reference_images=candidate_set['reference_images'], evaluation_images=candidate_set['evaluation_images'],
                    style_excluded_dimensions=['overall_style_characteristics'] + (['cut_style_characteristics', 'top_design_characteristics'] if kind == 'waist' else []),
                    eligibility_status='pending_human_same_garment_and_attribute_check; development_sources_not_blind_test'))
        w.save(out / 'candidate_task_bank.json', {'status': 'planning_only_not_formal_test', 'tasks': planned_tasks})
        with (out / 'human_attribute_check_nine.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image_id', 'path', 'dimension', 'visible_evidence', 'verified_same_garment_group', 'reviewer', 'status'])
            for candidate_set in candidates:
                for ref in candidate_set['reference_images'] + candidate_set['evaluation_images']:
                    for field in prompts['fields']:
                        writer.writerow([ref['image_id'], ref['path'], field, '', '', '', 'pending'])
        w.save(out / 'input_manifest.json', {'task': w.file_record(out / 'tasks.json'),
                                           'config': w.file_record(out / 'config.json'),
                                           'prompts': w.file_record(out / 'prompts.json')})
        print('Prepared one-task/one-repeat/six-condition probe; API calls=0')
        return
    if not a.execute_api:
        raise ValueError('Explicit API execution required')
    kit = Path(a.kit)
    cfg, task_file, prompts = w.load(kit / 'config.json'), w.load(kit / 'tasks.json'), w.load(kit / 'prompts.json')
    manifest = w.load(kit / 'input_manifest.json')
    for name, record in w.load(kit / 'code_snapshot.json').items():
        if w.digest(w.HERE / name) != record['sha256']:
            raise ValueError('Code changed after freezing; prepare a new kit')
    for key, filename in [('task', 'tasks.json'), ('config', 'config.json'), ('prompts', 'prompts.json')]:
        if w.digest(kit / filename) != manifest[key]['sha256']:
            raise ValueError('Frozen probe input changed')
    tasks = task_file['tasks']
    if len(tasks) != 1 or cfg['repeats'] != 1 or cfg['conditions'] != list(CONDITIONS):
        raise ValueError('Only the bounded six-output initial probe is enabled')
    if cfg['image_model'] != 'bytedance-seed/seedream-5-0-flash' or cfg['image_provider']['only'] != ['seed']:
        raise ValueError('Probe only permits the previously authorized generator route')
    w.validate_tasks(tasks, cfg)
    w.load_local_credentials()
    budget = Budget(out, a.budget_usd)
    w.save(out / 'frozen_config.json', cfg)
    w.save(out / 'frozen_tasks.json', tasks)
    w.save(out / 'frozen_prompts.json', prompts)
    w.save(out / 'condition_registry.json', CONDITIONS)
    task = tasks[0]
    refs = task['reference_images']
    paths = [r['path'] for r in refs]
    results = []
    stage = 'extraction'
    try:
        observations = []
        for ref in refs:
            print('Extracting reference:', ref['image_id'], flush=True)
            prompt = prompts['observe'].format(image_id=ref['image_id'], fields=','.join(prompts['fields']))
            obj = w.fields_check(w.strict_json(budget.call('text', cfg, prompt, [ref['path']])), prompts['fields'])
            observations.append({'image_id': ref['image_id'], 'fields': obj})
            w.save(out / 'observations.json', observations)
        profile = w.fields_check(w.strict_json(budget.call('text', cfg,
            prompts['aggregate'].format(fields=','.join(prompts['fields']), observations=json.dumps(observations, ensure_ascii=False)), [])), prompts['fields'])
        if any(v is not None and len(v)>60 for v in profile.values()):
            raise ValueError('Aggregate limit exceeded; no truncation or retry')
        summary = budget.call('text', cfg, prompts['free_summary'], paths)
        if len(summary)>480:
            raise ValueError('Summary limit exceeded; no truncation or retry')
        reconciliation = prompts['study_reconciliation'].format(summary=summary, explicit=json.dumps(task['explicit_fields'], ensure_ascii=False))
        reconciled = budget.call('text', cfg, reconciliation, [])
        if len(reconciled)>480:
            raise ValueError('Reconciled summary limit exceeded')
        w.save(out / 'profile.json', {'observations': observations, 'profile': profile, 'summary': summary,
                                    'reconciled_summary': reconciled, 'reference_records': [w.file_record(p) for p in paths]})
        scheduled = list(CONDITIONS)
        random.Random(cfg['schedule_seed']).shuffle(scheduled)
        w.save(out / 'schedule.json', scheduled)
        stage = 'generation_and_evaluation'
        for condition in scheduled:
            run_id = f"{task['task_id']}_{condition}_0"
            prompt = render(task, condition, profile, summary, reconciled, prompts)
            row = {'run_id': run_id, 'task_id': task['task_id'], 'condition': condition,
                   'condition_definition': CONDITIONS[condition], 'repeat': 0, 'prompt': prompt,
                   'prompt_sha256': w.text_hash(prompt), 'images': [w.file_record(p) for p in paths],
                   'seed': cfg['image_seed_base'], 'status': 'started', 'timestamp': w.stamp()}
            results.append(row)
            w.save(out / (run_id + '.json'), row)
            data = budget.call('image', cfg, prompt, paths, cfg['image_seed_base'])
            blob = w.output_bytes(data, cfg)
            with Image.open(io.BytesIO(blob)) as im:
                fmt, dimensions = im.format, im.size
                im.verify()
            dest = out / (run_id + {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[fmt])
            dest.write_bytes(blob)
            row.update(status='completed', output=w.file_record(dest), dimensions=dimensions, usage=data.get('usage'))
            w.save(out / (run_id + '.json'), row)
            w.save(out / 'results.json', results)
            judge_cfg = dict(cfg, vlm_model=cfg['judge_model'], vlm_extra={})
            constraints = [{'constraint_id': f'c{i+1}', 'text': value} for i,value in enumerate(task['hard_constraints'])]
            c_prompt = prompts['judge'].format(task=json.dumps({'brief': task['brief'], 'constraints': constraints}, ensure_ascii=False))
            answer = w.strict_json(budget.call('text', judge_cfg, c_prompt, [str(dest)]))
            scores = w.judge_checks(answer, [r['constraint_id'] for r in constraints])
            excluded = sorted(set(task['explicit_fields']) | set(task.get('style_excluded_dimensions', [])))
            eligible = [f for f in prompts['fields'] if f not in excluded]
            s_prompt = prompts['style_judge'].format(excluded_dimensions=json.dumps(excluded),
                                                    eligible_dimensions=json.dumps(eligible))
            style = w.strict_json(budget.call('text', judge_cfg, s_prompt, paths+[str(dest)]))
            style_scores = w.style_checks(style, eligible)
            w.save(out / (run_id+'.scores.json'), {'run_id': run_id, 'kind': 'unvalidated_machine_proxy',
                'raw_constraints': answer, 'raw_style': style,
                'constraint_metrics': scores, 'style_metrics': style_scores})
            w.save(out / 'status.json', {'status': 'running', 'generated': len(results), 'scored': len(list(out.glob('*.scores.json'))), 'planned': 6})
            print('Completed generation and two evaluations:', condition, flush=True)
        w.save(out / 'status.json', {'status': 'completed', 'generated': 6, 'scored': 6, 'planned': 6,
                    'interpretation': 'One exploratory task cannot establish method superiority or nine-dimension necessity'})
    except Exception as exc:
        if results and results[-1]['status'] != 'completed':
            results[-1].update(status='failed', error_type=type(exc).__name__)
            w.save(out / (results[-1]['run_id']+'.json'), results[-1])
        w.save(out / 'results.json', results)
        w.save(out / 'status.json', {'status': 'stopped', 'stage': stage, 'error_type': type(exc).__name__,
                    'generated': sum(r['status']=='completed' for r in results), 'no_automatic_retry': True})
        raise


if __name__ == '__main__':
    main()
