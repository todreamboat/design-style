"""Consented 24-output format ablation. Persistent budget, no automatic retries."""
import argparse
import io
import json
import random
from pathlib import Path
from PIL import Image
import workflow_benchmark as w
from nine_dimension_pilot import Budget
import p1_supplement as p


class PersistentBudget(Budget):
    def __init__(self, out, cap):
        super().__init__(out, cap)
        ledger = out/'budget_ledger.json'
        if ledger.exists():
            saved = w.load(ledger)
            if saved['cap_usd'] != cap:
                raise ValueError('Cannot change consented budget on resume')
            self.rows = saved['requests']
        # In-flight requests keep the conservative reservation after a crash.


def run(cap):
    if not 0 < cap <= 5:
        raise ValueError('User consent cap: 5 USD')
    protocol = w.load(p.OUT/'ablation_protocol.json')
    cfg, prompts = protocol['config'], protocol['prompts']
    if cfg['backend'] != 'openrouter' or cfg['image_model'] != 'bytedance-seed/seedream-5-0-flash':
        raise ValueError('Generator route changed')
    if cfg['image_provider']['only'] != ['seed'] or cfg['judge_model'] != 'x-ai/grok-4.7':
        raise ValueError('Frozen model route changed')
    plans = protocol['planned_runs']
    if len(plans) != 24:
        raise ValueError('Expected exactly 24 paired generations')
    for plan in plans:
        if w.text_hash(plan['prompt']) != plan['prompt_sha256']:
            raise ValueError('Prompt hash mismatch')
        if w.digest(p.ROOT/'seed'/(plan['source_id']+'.profile.json')) != plan['profile_sha256']:
            raise ValueError('Profile changed')
        for ref in plan['reference_images']:
            if w.digest(ref['path']) != ref['sha256']:
                raise ValueError('Reference changed')
    out = p.OUT/'flat_base'
    out.mkdir(exist_ok=True)
    frozen = out/'frozen_protocol.json'
    if frozen.exists() and w.load(frozen) != protocol:
        raise ValueError('Protocol changed after dispatch')
    w.save(frozen, protocol)
    tasks = {t['task_id']:t for t in protocol['tasks']}
    order = list(plans)
    random.Random(20261012).shuffle(order)
    w.save(out/'schedule.json', [[r['task_id'],r['condition'],r['repeat']] for r in order])
    w.load_local_credentials()
    budget = PersistentBudget(out, cap)
    judge_cfg = dict(cfg, vlm_model=cfg['judge_model'], vlm_extra={})
    def status(state, **extra):
        w.save(out/'status.json',dict(status=state, planned=24,
            generated=sum(w.load(f).get('status')=='completed' for f in out.glob('main_*.json') if not f.name.endswith('.scores.json')),
            scored=len(list(out.glob('*.scores.json'))), **extra))
    status('running', authorization='user_consented_5usd_20261012', human_calibration=False)
    try:
        for plan in order:
            task = tasks[plan['task_id']]
            run_id = f"{plan['task_id']}_D_base_{plan['repeat']}"
            record_path = out/(run_id+'.json')
            score_path = out/(run_id+'.scores.json')
            if score_path.exists():
                record = w.load(record_path)
                if w.digest(record['output']['path']) != record['output']['sha256']:
                    raise ValueError('Saved output changed')
                continue
            paths = [r['path'] for r in plan['reference_images']]
            record = w.load(record_path) if record_path.exists() else None
            if not record or record.get('status') != 'completed':
                # Re-execution after a failed dispatch must be explicitly requested.
                if record is not None:
                    last_call=out/'calls'/f'request_{len(budget.rows):03d}'/'call_00001.error.json'
                    local_denial = last_call.exists() and 'PermissionError' in w.load(last_call).get('exception_chain_types',[])
                    if not local_denial:
                        raise ValueError('Prior failed/in-flight image dispatch: inspect before retry')
                record = dict(plan, run_id=run_id, backend='seed', status='started')
                w.save(record_path, record)
                print('Generating',run_id,flush=True)
                response = budget.call('image',cfg,plan['prompt'],paths,plan['image_seed'])
                blob = w.output_bytes(response,cfg)
                with Image.open(io.BytesIO(blob)) as image:
                    fmt, dimensions = image.format,image.size
                    image.verify()
                dest = out/(run_id+{'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
                dest.write_bytes(blob)
                record.update(status='completed',output=w.file_record(dest),dimensions=dimensions)
                w.save(record_path,record)
            dest = record['output']['path']
            checks = [{'constraint_id':f'{prefix}{i+1}','text':value}
                for prefix,key in [('t','hard_constraints'),('p','presentation_checks'),('q','artifact_checks')]
                for i,value in enumerate(task[key])]
            constraints_file = out/(run_id+'.constraints.json')
            if constraints_file.exists():
                raw = w.load(constraints_file)
            else:
                raw = w.strict_json(budget.call('text',judge_cfg,prompts['judge'].format(
                    task=json.dumps({'brief':task['brief'],'constraints':checks},ensure_ascii=False)),[dest]))
                w.judge_checks(raw,[c['constraint_id'] for c in checks])
                w.save(constraints_file,raw)
            metrics = {prefix:w.judge_checks({'checks':[r for r in raw['checks'] if r['constraint_id'].startswith(prefix)]},
                [c['constraint_id'] for c in checks if c['constraint_id'].startswith(prefix)]) for prefix in ('t','p','q')}
            excluded = sorted(set(task['explicit_fields'])|set(task['style_excluded_dimensions']))
            eligible = [f for f in prompts['fields'] if f not in excluded]
            style = w.strict_json(budget.call('text',judge_cfg,prompts['style_judge'].format(
                excluded_dimensions=json.dumps(excluded),eligible_dimensions=json.dumps(eligible)),paths+[dest]))
            w.save(score_path,dict(run_id=run_id,task_id=task['task_id'],source_id=task['source_id'],
                condition='D_base',repeat=plan['repeat'],backend='seed',raw_constraints=raw,raw_style=style,
                task_metrics=metrics['t'],presentation_metrics=metrics['p'],artifact_diagnostics=metrics['q'],
                style_metrics=w.style_checks(style,eligible),kind='uncalibrated_machine_proxy'))
            status('running')
            print('Scored',len(list(out.glob('*.scores.json'))),'/24; accounted USD',
                round(sum(r['charged_or_reserved_usd'] for r in budget.rows),4),flush=True)
        status('completed',formal_test=False,human_calibration=False)
    except Exception as exc:
        status('stopped',error_type=type(exc).__name__,no_automatic_retry=True)
        print('Stopped:',type(exc).__name__,str(exc)[:200],flush=True)
        raise SystemExit(1) from None


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-api',action='store_true')
    parser.add_argument('--budget-usd',type=float,default=5)
    args=parser.parse_args()
    if not args.execute_api:
        parser.error('Explicit --execute-api required')
    run(args.budget_usd)
