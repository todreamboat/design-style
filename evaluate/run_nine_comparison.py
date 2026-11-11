"""Authorized Seed + FLUX comparison, one shared budget, no automatic retry.

Original frozen main study and tasks stay immutable. This release adds orchestration,
public FLUX capability verification and its separately identified model-family arm.
"""
import argparse
import io
import random
import time
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import workflow_benchmark as w
import nine_main_study as study

FLUX_ID='black-forest-labs/flux.2-klein-4b'
FLUX_PUBLIC='https://openrouter.ai/api/v1/images/models/'+FLUX_ID+'/endpoints'

def verify_flux(data):
    if data.get('id') != FLUX_ID:
        raise ValueError('Unexpected FLUX model')
    eps=[e for e in data.get('endpoints',[]) if e.get('provider_slug')=='black-forest-labs']
    if len(eps)!=1:
        raise ValueError('Expected one authorized Black Forest Labs endpoint')
    params=eps[0]['supported_parameters']
    if '2:3' not in params['aspect_ratio']['values'] or params['input_references']['max']<2 or params['n']['max']!=1 or 'seed' not in params:
        raise ValueError('FLUX endpoint lacks frozen required capabilities')
    return eps[0]

class JointBudget(study.StudyBudget):
    root=None
    shared_rows=None
    started=None
    total_cap=None

    def __init__(self, stage_out, cap):
        if not 0<cap<=20 or cap!=type(self).total_cap:
            raise ValueError('Joint authorized budget is at most 20 USD')
        self.stage_out=Path(stage_out)
        self.stage=self.stage_out.name
        self.out=type(self).root/'budget'
        self.cap=cap
        self.rows=type(self).shared_rows

    def call(self,kind,cfg,prompt,paths,seed=None):
        if time.monotonic()-type(self).started>6*3600:
            raise ValueError('Six-hour dispatch deadline reached; no further calls')
        before=len(self.rows)
        try:
            return super().call(kind,cfg,prompt,paths,seed)
        finally:
            if len(self.rows)>before:
                self.rows[-1]['stage']=self.stage
            w.save(self.out/'budget_ledger.json',{'cap_usd':self.cap,'requests':self.rows,
                'scope':'All Seed and FLUX stages combined; unknown usage retains reservation; not provider-enforced ceiling'})
            w.save(self.stage_out/'budget_ledger.json',{'cap_usd':self.cap,
                'requests':[r for r in self.rows if r.get('stage')==self.stage],
                'joint_ledger_path':str(self.out/'budget_ledger.json')})

def evaluate_one(task,run_id,condition,repeat,dest,paths,cfg,prompts,budget,out):
    judge_cfg=dict(cfg,vlm_model=cfg['judge_model'],vlm_extra={})
    checks=[{'constraint_id':f'{prefix}{i+1}','text':value}
            for prefix,key in [('t','hard_constraints'),('p','presentation_checks'),('q','artifact_checks')]
            for i,value in enumerate(task[key])]
    raw=w.strict_json(budget.call('text',judge_cfg,prompts['judge'].format(
        task=__import__('json').dumps({'brief':task['brief'],'constraints':checks},ensure_ascii=False)),[str(dest)]))
    w.judge_checks(raw,[c['constraint_id'] for c in checks])
    metrics={prefix:w.judge_checks({'checks':[r for r in raw['checks'] if r['constraint_id'].startswith(prefix)]},
        [c['constraint_id'] for c in checks if c['constraint_id'].startswith(prefix)]) for prefix in ('t','p','q')}
    excluded=sorted(set(task['explicit_fields'])|set(task['style_excluded_dimensions']))
    eligible=[f for f in prompts['fields'] if f not in excluded]
    style=w.strict_json(budget.call('text',judge_cfg,prompts['style_judge'].format(
        excluded_dimensions=__import__('json').dumps(excluded),eligible_dimensions=__import__('json').dumps(eligible)),paths+[str(dest)]))
    w.save(out/(run_id+'.scores.json'),dict(run_id=run_id,task_id=task['task_id'],source_id=task['source_id'],
        condition=condition,repeat=repeat,backend='flux_klein_4b',raw_constraints=raw,raw_style=style,
        task_metrics=metrics['t'],presentation_metrics=metrics['p'],artifact_diagnostics=metrics['q'],
        style_metrics=w.style_checks(style,eligible),kind='uncalibrated_machine_proxy'))

def flux_run(kit,seed_out,out,cap):
    cfg=w.load(kit/'flux_candidate_config.json')
    if cfg['image_model']!=FLUX_ID or cfg['image_provider']['only']!=['black-forest-labs']:
        raise ValueError('FLUX route is not the authorized route')
    cfg.update(authorization_status='user_approved_20261006',parameter_support_status='public_endpoint_verified')
    tasks=w.load(kit/'tasks.json')['tasks']
    prompts=w.load(kit/'prompts.json')
    out.mkdir(parents=True,exist_ok=False)
    budget=JointBudget(out,cap)
    for name,obj in [('frozen_config.json',cfg),('frozen_tasks.json',tasks),('frozen_prompts.json',prompts),
                     ('review_gate.json',w.load(kit/'review_gate.json'))]:
        w.save(out/name,obj)
    schedule=[(t['task_id'],c,r) for t in tasks for c in ('A','C') for r in range(cfg['repeats'])]
    random.Random(cfg['schedule_seed']).shuffle(schedule)
    w.save(out/'schedule.json',schedule)
    by_task={t['task_id']:t for t in tasks}
    rows=[]
    try:
        for task_id,condition,repeat in schedule:
            task=by_task[task_id]
            profile=w.load(seed_out/(task['source_id']+'.profile.json'))
            desc=w.load(seed_out/(task_id+'.descriptions.json'))
            paths=[r['path'] for r in task['reference_images']]
            prompt=study.render(task,condition,profile['profile'],profile['summary'],desc['reconciled'],prompts,desc['literature'])
            run_id=f'{task_id}_{condition}_{repeat}'
            row=dict(run_id=run_id,task_id=task_id,source_id=task['source_id'],condition=condition,repeat=repeat,
                backend='flux_klein_4b',status='started',prompt=prompt,prompt_sha256=w.text_hash(prompt),
                seed=cfg['image_seed_base']+repeat,input_records=[w.file_record(p) for p in paths])
            rows.append(row)
            w.save(out/(run_id+'.json'),row)
            image=budget.call('image',cfg,prompt,paths,row['seed'])
            blob=w.output_bytes(image,cfg)
            with Image.open(io.BytesIO(blob)) as im:
                fmt,dimensions=im.format,im.size
                im.verify()
            dest=out/(run_id+{'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
            dest.write_bytes(blob)
            row.update(status='completed',output=w.file_record(dest),dimensions=dimensions)
            w.save(out/(run_id+'.json'),row)
            evaluate_one(task,run_id,condition,repeat,dest,paths,cfg,prompts,budget,out)
            w.save(out/'results.json',rows)
            w.save(out/'status.json',dict(status='running',generated=len(rows),scored=len(list(out.glob('*.scores.json'))),planned=48))
            print('FLUX completed',len(rows),'/48',run_id,flush=True)
        w.save(out/'status.json',dict(status='completed',generated=48,scored=48,formal_test=False,
            interpretation='Model-family adaptation; not Dress-EM or Dress-ED benchmark reproduction'))
    except Exception as exc:
        if rows and rows[-1]['status']=='started':
            rows[-1].update(status='failed',error_type=type(exc).__name__)
            w.save(out/(rows[-1]['run_id']+'.json'),rows[-1])
        w.save(out/'results.json',rows)
        w.save(out/'status.json',dict(status='stopped',generated=sum(r['status']=='completed' for r in rows),
            scored=len(list(out.glob('*.scores.json'))),error_type=type(exc).__name__,no_automatic_retry=True))
        raise

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kit',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--budget-usd',type=float,default=20)
    p.add_argument('--execute-api',action='store_true')
    p.add_argument('--authorize-bfl-upload',action='store_true')
    p.add_argument('--allow-unreviewed-development',action='store_true')
    a=p.parse_args()
    if not a.execute_api or not a.authorize_bfl_upload or not 0<a.budget_usd<=20:
        raise ValueError('Explicit execution, BFL image-upload authorization and total budget required')
    kit,out=Path(a.kit).resolve(),Path(a.out).resolve()
    if (out/'release.json').exists():
        raise ValueError('Never resume or retry an existing comparison release automatically')
    out.mkdir(parents=True,exist_ok=True)
    for name,record in w.load(kit/'input_manifest.json').items():
        if w.digest(kit/name)!=record['sha256']:
            raise ValueError('Frozen input changed')
    for name,record in w.load(kit/'code_snapshot.json').items():
        if w.digest(w.HERE/name)!=record['sha256']:
            raise ValueError('Frozen study code changed')
    w.save(out/'release.json',dict(date='2026-10-06',timestamp_utc=w.stamp(),cap_usd=a.budget_usd,
        user_authorized_bfl_upload=True,content='Frozen reference images, generated images and task prompts only; no manuscript',
        planned_seed_images=192,planned_flux_images=48,dispatch_deadline_hours=6,
        original_kit_manifest=w.file_record(kit/'input_manifest.json'),
        wrapper_code=w.file_record(Path(__file__)),flux_candidate_config=w.file_record(kit/'flux_candidate_config.json'),
        review_gate=w.load(kit/'review_gate.json')))
    JointBudget.root=out
    JointBudget.shared_rows=[]
    JointBudget.total_cap=a.budget_usd
    JointBudget.started=time.monotonic()
    stage='public_flux_capability_verification'
    try:
        import requests
        response=requests.get(FLUX_PUBLIC,timeout=(15,30))
        response.raise_for_status()
        data=response.json()
        verify_flux(data)
        w.save(out/'flux_public_endpoint.json',data)
        stage='seed_main'
        w.save(out/'status.json',dict(status='running',stage=stage,cap_usd=a.budget_usd,planned_images=240))
        study.StudyBudget=JointBudget
        study.run(SimpleNamespace(execute_api=True,kit=str(kit),out=str(out/'seed'),
            budget_usd=a.budget_usd,allow_unreviewed_development=a.allow_unreviewed_development))
        stage='flux_comparison'
        w.save(out/'status.json',dict(status='running',stage=stage,completed_seed_images=192,planned_flux_images=48))
        flux_run(kit,out/'seed',out/'flux',a.budget_usd)
        w.save(out/'status.json',dict(status='completed',seed_images=192,flux_images=48,scored_images=240,
            formal_test=False,human_review_complete=False,reported_usage_cost_usd=sum(r.get('returned_cost_usd',0) for r in JointBudget.shared_rows)))
        print('Joint comparison completed; descriptive summaries ready to run',flush=True)
    except Exception as exc:
        w.save(out/'status.json',dict(status='stopped',stage=stage,error_type=type(exc).__name__,
            no_automatic_retry=True,reported_usage_cost_usd=sum(r.get('returned_cost_usd',0) for r in JointBudget.shared_rows)))
        print('Stopped at',stage,type(exc).__name__,'without automatic retry',flush=True)
        raise RuntimeError('Comparison stopped; see retained stage and request metadata') from None

if __name__=='__main__':
    main()
