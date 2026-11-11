"""Explicitly requested continuation; preserves frozen code and paid results."""
import argparse
import io
import json
import os
from PIL import Image
import run_saved_style as s

w = s.w
OUT = s.OUT


def evaluate(task, plan, dest, refs, cfg, prompts, budget):
    """Keep valid judgments; repeat malformed responses at most once, unchanged."""
    judge_cfg = dict(cfg, vlm_model=cfg['judge_model'], vlm_extra={})
    checks = [{'constraint_id':f'{prefix}{i+1}','text':value}
              for prefix,key in [('t','hard_constraints'),('p','presentation_checks'),('q','artifact_checks')]
              for i,value in enumerate(task[key])]
    excluded = sorted(set(task['explicit_fields']) | set(task['style_excluded_dimensions']))
    eligible = [f for f in prompts['fields'] if f not in excluded]
    def judged(label, prompt, paths, validate):
        cache = OUT/(plan['run_id']+'.'+label+'.json')
        if cache.exists():
            raw = w.load(cache)
            validate(raw)
            return raw
        for attempt in range(2):
            text = budget.call('text',judge_cfg,prompt,paths)
            try:
                raw = w.strict_json(text)
                validate(raw)
            except (ValueError,KeyError,TypeError):
                if attempt:
                    raise
                print('Malformed',label,'response; one unchanged evaluation retry',flush=True)
                continue
            w.save(cache,raw)
            return raw
    raw = judged('constraints',prompts['judge'].format(
        task=json.dumps({'brief':task['brief'],'constraints':checks},ensure_ascii=False)),[str(dest)],
        lambda answer:w.judge_checks(answer,[c['constraint_id'] for c in checks]))
    metrics = {prefix:w.judge_checks({'checks':[r for r in raw['checks'] if r['constraint_id'].startswith(prefix)]},
        [c['constraint_id'] for c in checks if c['constraint_id'].startswith(prefix)]) for prefix in ('t','p','q')}
    style = judged('style',prompts['style_judge'].format(
        excluded_dimensions=json.dumps(excluded),eligible_dimensions=json.dumps(eligible)),refs+[str(dest)],
        lambda answer:w.style_checks(answer,eligible))
    w.save(OUT/(plan['run_id']+'.scores.json'),dict(run_id=plan['run_id'],task_id=task['task_id'],
        source_id=task['source_id'],condition=plan['condition'],repeat=plan['repeat'],backend='seed_text_only',
        raw_constraints=raw,raw_style=style,task_metrics=metrics['t'],presentation_metrics=metrics['p'],
        artifact_diagnostics=metrics['q'],style_metrics=w.style_checks(style,eligible),kind='uncalibrated_machine_proxy'))


def run():
    protocol = w.load(OUT/'protocol.json')
    old_status = w.load(OUT/'status.json')
    if old_status['status'] != 'stopped':
        raise ValueError('Only a stopped experiment may resume')
    try:
        os.kill(old_status['pid'], 0)
    except OSError:
        pass
    else:
        raise ValueError('Previous PID still exists; inspect before resume')
    for ref in list(protocol['saved_inputs'].values()) + list(protocol['code_inputs'].values()):
        if w.digest(ref['path']) != ref['sha256']:
            raise ValueError('Frozen code or saved description changed')
    for task in protocol['tasks']:
        for ref in task['reference_images']:
            if w.digest(ref['path']) != ref['sha256']:
                raise ValueError('Evaluation reference changed')
    cfg, prompts = protocol['config'], protocol['prompts']
    if cfg['image_model'] != 'bytedance-seed/seedream-5-0-flash' or cfg['judge_model'] != 'x-ai/grok-4.7':
        raise ValueError('Frozen route changed')
    by_task = {t['task_id']:t for t in protocol['tasks']}
    # Validate every completed image before any paid dispatch.
    for plan in protocol['plans']:
        if w.text_hash(plan['prompt']) != plan['prompt_sha256'] or plan['input_records']:
            raise ValueError('Prompt changed or generation image leakage')
        path = OUT/(plan['run_id']+'.json')
        if path.exists():
            record = w.load(path)
            if record.get('status') == 'completed':
                if w.digest(record['output']['path']) != record['output']['sha256']:
                    raise ValueError('Completed image changed')
            if (OUT/(plan['run_id']+'.scores.json')).exists() and record.get('status') != 'completed':
                raise ValueError('Score without completed image')
    w.load_local_credentials()
    budget = s.PersistentBudget(OUT, 5)
    started = w.stamp()
    history = old_status.get('session_history', []) + [old_status]
    # Avoid recursively embedding prior history.
    history[-1] = {k:v for k,v in old_status.items() if k != 'session_history'}
    def status(state, **extra):
        w.save(OUT/'status.json', dict(status=state,pid=os.getpid(),started_utc=started,
            planned=72, generated=sum(w.load(f).get('status') == 'completed'
                for f in OUT.glob('main_*.json') if not f.name.endswith('.scores.json')),
            scored=len(list(OUT.glob('*.scores.json'))),
            accounted_usd=sum(r['charged_or_reserved_usd'] for r in budget.rows),
            session_history=history, no_automatic_retry=True, **extra))
    status('running')
    try:
        for plan in protocol['plans']:
            record_path = OUT/(plan['run_id']+'.json')
            score_path = OUT/(plan['run_id']+'.scores.json')
            if score_path.exists():
                continue
            record = w.load(record_path) if record_path.exists() else None
            if not record or record.get('status') != 'completed':
                attempts = ([record] if record else [])
                record = dict(plan,status='started',backend='seed_text_only',
                              previous_attempts=attempts,resume_authorization='user_requested_continue')
                w.save(record_path,record)
                print('Generating',plan['run_id'],flush=True)
                response = s.image_call(budget,cfg,plan)
                blob = w.output_bytes(response,cfg)
                with Image.open(io.BytesIO(blob)) as image:
                    fmt,dimensions = image.format,image.size
                    image.verify()
                dest = OUT/(plan['run_id']+{'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
                dest.write_bytes(blob)
                record.update(status='completed',output=w.file_record(dest),dimensions=dimensions)
                w.save(record_path,record)
            dest = record['output']['path']
            task = dict(by_task[plan['task_id']],brief=s.BRIEFS[plan['task_id'].rsplit('_',1)[1]])
            refs = [r['path'] for r in task['reference_images']]
            evaluate(task,plan,dest,refs,cfg,prompts,budget)
            score = w.load(score_path)
            score['backend'] = 'seed_text_only'
            w.save(score_path,score)
            status('running')
            print('Scored',len(list(OUT.glob('*.scores.json'))),'/72; accounted USD',
                  round(sum(r['charged_or_reserved_usd'] for r in budget.rows),4),flush=True)
        status('completed')
    except Exception as exc:
        status('stopped',error_type=type(exc).__name__)
        print('Stopped:',type(exc).__name__,str(exc)[:200],flush=True)
        raise SystemExit(1) from None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-api',action='store_true')
    args = parser.parse_args()
    if not args.execute_api:
        parser.error('Explicit --execute-api required')
    run()
