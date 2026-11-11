"""Consented text-only saved-style experiment: 72 outputs, no image inputs.

Historical images are used only by the style evaluator. No automatic retries.
"""
import argparse
import io
import json
import os
import random
import time
from pathlib import Path
from PIL import Image
import workflow_benchmark as w
import p1_supplement as p
from run_p1_ablation import PersistentBudget
from run_nine_comparison import evaluate_one

OUT = p.ROOT / 'saved_style_20261014'
CONDITIONS = ('task_only', 'saved_summary', 'saved_nine')
BRIEFS = {
    'appearance': '设计一件外套：外套外表面整体为紫色，具有清晰连续的白色竖条纹；内搭和下装不要求改色。',
    'structure': '设计一件外套：腰部使用一条可见腰带明确收束；领口左右对称。',
}
COMMON = ('完成服装概念设计任务：{brief}\n必须满足：{constraints}\n'
          '当前明确任务要求优先于历史风格描述。未指定的服装属性可参考下述描述，'
          '没有描述时自行设计。没有输入图片，不需要复制某张作品。'
          '采用统一浅灰背景，由中性模特全身正面穿着展示完整服装，'
          '使外套下摆与腿部位置清晰可判断。只输出一张图，不加文字、水印或拼图。\n'
          '已保存的历史服装风格描述（null表示无附加要求）：{content}')


def render(task, condition, saved):
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    if task.get('input_image'):
        raise ValueError('Editing images are excluded from this experiment')
    content = {'task_only': '无', 'saved_summary': saved['summary'],
               'saved_nine': json.dumps(saved['profile'], ensure_ascii=False)}[condition]
    return COMMON.format(brief=BRIEFS[task['task_id'].rsplit('_', 1)[1]],
        constraints=json.dumps(task['hard_constraints'], ensure_ascii=False), content=content)


def prepare():
    if (OUT / 'protocol.json').exists():
        raise ValueError('Existing frozen protocol must not be overwritten')
    cfg = w.load(p.KIT / 'config.json')
    prompts = w.load(p.KIT / 'prompts.json')
    tasks = w.load(p.KIT / 'tasks.json')['tasks']
    assert len(tasks) == 12 and len({t['source_id'] for t in tasks}) == 6
    plans = []
    inputs = {}
    for source_index, source in enumerate(sorted({t['source_id'] for t in tasks})):
        path = p.ROOT / 'seed' / (source + '.profile.json')
        saved = w.load(path)
        assert set(saved['profile']) == set(prompts['fields'])
        assert isinstance(saved['summary'], str) and saved['summary'].strip()
        inputs[source] = w.file_record(path)
        for task in [t for t in tasks if t['source_id'] == source]:
            for condition in CONDITIONS:
                for repeat in range(2):
                    prompt = render(task, condition, saved)
                    plans.append(dict(run_id=f"{task['task_id']}_{condition}_{repeat}",
                        task_id=task['task_id'], source_id=source, condition=condition,
                        repeat=repeat, prompt=prompt, prompt_sha256=w.text_hash(prompt),
                        seed=20261014 + 2 * source_index + repeat, input_records=[]))
    random.Random(20261014).shuffle(plans)
    protocol = dict(version='saved-style-v1-20261014', config=cfg, prompts=prompts,
        tasks=tasks, plans=plans, saved_inputs=inputs, budget_cap_usd=5,
        authorization='user_consented_72_outputs_5usd', generation_image_inputs=0,
        evaluation_reference_images='original two extraction images, not held-out style generalization',
        independent_units=6, aggregation='repeat within task, then task within source',
        analysis='primary saved_nine minus task_only and saved_summary; task and style endpoints; four-test Holm',
        interpretation='new task reuse, not longitudinal or cross-series validation; machine attribute proxy',
        shared_prompt_template=COMMON, briefs=BRIEFS,
        code_inputs={name:w.file_record(w.HERE/name) for name in
                     ('run_saved_style.py','workflow_benchmark.py','run_nine_comparison.py',
                      'run_p1_ablation.py','nine_dimension_pilot.py')})
    w.save(OUT / 'protocol.json', protocol)
    print('Prepared 72 frozen requests; generation image inputs=0; API calls=0', flush=True)


def image_call(budget, cfg, plan):
    if plan['input_records']:
        raise ValueError('Historical images must never enter generation')
    return budget.call('image', cfg, plan['prompt'], [], plan['seed'])


def run():
    protocol = w.load(OUT / 'protocol.json')
    cfg, prompts = protocol['config'], protocol['prompts']
    if cfg['image_model'] != 'bytedance-seed/seedream-5-0-flash' or cfg['judge_model'] != 'x-ai/grok-4.7':
        raise ValueError('Consented model route changed')
    if cfg['backend'] != 'openrouter' or cfg['image_provider']['only'] != ['seed']:
        raise ValueError('Consented provider changed')
    for ref in list(protocol['saved_inputs'].values()) + list(protocol['code_inputs'].values()):
        if w.digest(ref['path']) != ref['sha256']:
            raise ValueError('Frozen input or code changed')
    for task in protocol['tasks']:
        for ref in task['reference_images']:
            if w.digest(ref['path']) != ref['sha256']:
                raise ValueError('Evaluation reference changed')
    if (OUT / 'status.json').exists():
        raise ValueError('Prior dispatch exists; inspect and obtain consent before resume')
    w.load_local_credentials()
    budget = PersistentBudget(OUT, 5)
    by_task = {t['task_id']:t for t in protocol['tasks']}
    start = time.monotonic()
    def status(state, **extra):
        w.save(OUT/'status.json', dict(status=state, pid=os.getpid(), started_utc=started,
            planned=72, generated=sum(w.load(f).get('status') == 'completed'
                for f in OUT.glob('main_*.json') if not f.name.endswith('.scores.json')),
            scored=len(list(OUT.glob('*.scores.json'))),
            accounted_usd=sum(r['charged_or_reserved_usd'] for r in budget.rows), **extra))
    started = w.stamp()
    status('running')
    try:
        for plan in protocol['plans']:
            if time.monotonic() - start > 1800:
                raise TimeoutError('30-minute dispatch deadline reached')
            if w.text_hash(plan['prompt']) != plan['prompt_sha256']:
                raise ValueError('Frozen prompt changed')
            record = dict(plan, status='started', backend='seed_text_only')
            record_path = OUT/(plan['run_id']+'.json')
            w.save(record_path, record)
            print('Generating', plan['run_id'], flush=True)
            response = image_call(budget, cfg, plan)
            blob = w.output_bytes(response, cfg)
            with Image.open(io.BytesIO(blob)) as image:
                fmt, dimensions = image.format, image.size
                image.verify()
            dest = OUT/(plan['run_id']+{'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
            dest.write_bytes(blob)
            record.update(status='completed', output=w.file_record(dest), dimensions=dimensions)
            w.save(record_path, record)
            task = dict(by_task[plan['task_id']], brief=BRIEFS[plan['task_id'].rsplit('_',1)[1]])
            refs = [r['path'] for r in task['reference_images']]
            evaluate_one(task, plan['run_id'], plan['condition'], plan['repeat'],
                         dest, refs, cfg, prompts, budget, OUT)
            # The reused evaluator's backend name is historical; correct metadata only.
            score_path = OUT/(plan['run_id']+'.scores.json')
            scores = w.load(score_path)
            scores['backend'] = 'seed_text_only'
            w.save(score_path, scores)
            status('running')
            print('Scored', len(list(OUT.glob('*.scores.json'))), '/72; accounted USD',
                  round(sum(r['charged_or_reserved_usd'] for r in budget.rows),4), flush=True)
        status('completed')
    except Exception as exc:
        status('stopped', error_type=type(exc).__name__, no_automatic_retry=True)
        print('Stopped:',type(exc).__name__,str(exc)[:200],flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run'])
    parser.add_argument('--execute-api', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.execute_api:
        run()
    else:
        parser.error('Explicit --execute-api required')
