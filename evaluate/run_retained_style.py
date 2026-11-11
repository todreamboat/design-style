"""Six re-aggregations + 24 paired images; 3 USD, no original image generation input."""
import argparse
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import run_saved_style as s
import resume_saved_style as evaluator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.style_schema import synthesis_prompt, validate_synthesis, AGGREGATION_VERSION
from core.style_analyzer import StyleAnalyzer

w = s.w
OUT = s.p.ROOT/'retained_style_20261015'


def prepare():
    if (OUT/'protocol.json').exists():
        raise ValueError('Protocol already frozen')
    old = w.load(s.OUT/'protocol.json')
    sources = {}
    for source, record in old['saved_inputs'].items():
        if w.digest(record['path']) != record['sha256']:
            raise ValueError('Historical profile changed')
        saved = w.load(record['path'])
        analyses = [json.dumps(o['fields'],ensure_ascii=False) for o in saved['observations']]
        prompt = synthesis_prompt(analyses).replace('English', 'Chinese')
        sources[source] = dict(original=record,analyses=analyses,prompt=prompt,
            prompt_sha256=w.text_hash(prompt),image_ids=[o['image_id'] for o in saved['observations']])
    plans = [dict(p,run_id=p['run_id'].replace('_saved_nine_','_saved_nine_retained_'),
                  condition='saved_nine_retained',paired_existing_run=p['run_id'])
             for p in old['plans'] if p['condition']=='saved_nine']
    for p in plans:
        p.pop('prompt')
        p.pop('prompt_sha256')
    assert len(plans)==24 and len(sources)==6
    code_names = ['run_retained_style.py','run_saved_style.py','resume_saved_style.py',
                  'workflow_benchmark.py','run_p1_ablation.py','nine_dimension_pilot.py']
    code = [w.file_record(w.HERE/n) for n in code_names]
    code += [w.file_record(w.HERE.parent/'core'/n) for n in ('style_schema.py','style_analyzer.py')]
    w.save(OUT/'protocol.json',dict(version=AGGREGATION_VERSION,sources=sources,plans=plans,
        config=old['config'],prompts=old['prompts'],tasks=old['tasks'],code_inputs=code,
        cap_usd=3,authorization='user_consented_additional_3usd_20261015',
        scope='6 text-only aggregations, 24 images and paired machine evaluations',
        changes='aggregation policy and output length; original observations and Chinese output retained',
        generation_wrapper='original run_saved_style.render unchanged; zero generation image inputs',
        limitations=['post-result development adaptation, not held-out confirmation',
                     'historical comparators in earlier API batch; service drift cannot be excluded',
                     'aggregation is stochastic; not isolated wording or length effect'],
        old_protocol=w.file_record(s.OUT/'protocol.json')))
    print('Prepared; 6 aggregations + 24 images; no API calls',flush=True)


def run():
    protocol = w.load(OUT/'protocol.json')
    if (OUT/'status.json').exists():
        raise ValueError('Prior dispatch exists; inspect before explicitly continuing')
    for record in protocol['code_inputs']+[protocol['old_protocol']]+[
            x['original'] for x in protocol['sources'].values()]:
        if w.digest(record['path']) != record['sha256']:
            raise ValueError('Frozen input changed')
    cfg,prompts = protocol['config'],protocol['prompts']
    if cfg['backend']!='openrouter' or cfg['image_model']!='bytedance-seed/seedream-5-0-flash':
        raise ValueError('Generator route changed')
    if cfg['vlm_model']!='qwen/qwen3.8-max-0902' or cfg['judge_model']!='x-ai/grok-4.7':
        raise ValueError('Text route changed')
    for task in protocol['tasks']:
        for record in task['reference_images']:
            if w.digest(record['path'])!=record['sha256']:
                raise ValueError('Evaluation reference changed')
    w.load_local_credentials()
    budget=s.PersistentBudget(OUT,3)
    tasks={t['task_id']:t for t in protocol['tasks']}
    started=w.stamp()
    def status(state,**extra):
        w.save(OUT/'status.json',dict(status=state,pid=os.getpid(),started_utc=started,
            profiles=len(list(OUT.glob('*/train.json'))),planned=24,
            generated=sum(w.load(OUT/(p['run_id']+'.json')).get('status')=='completed'
                          for p in protocol['plans'] if (OUT/(p['run_id']+'.json')).exists()),
            scored=len(list(OUT.glob('*.scores.json'))),
            accounted_usd=sum(r['charged_or_reserved_usd'] for r in budget.rows),**extra))
    status('running')
    try:
        analyzer=object.__new__(StyleAnalyzer)
        analyzer.vlm_client=SimpleNamespace(model_name=cfg['vlm_model'])
        analyzer.llm_client=SimpleNamespace(model_name=cfg['vlm_model'])
        for source,data in sorted(protocol['sources'].items()):
            print('Aggregating',source,flush=True)
            if w.text_hash(data['prompt'])!=data['prompt_sha256']:
                raise ValueError('Aggregation prompt changed')
            raw=budget.call('text',cfg,data['prompt'],[])
            summary=validate_synthesis(w.strict_json(raw),data['analyses'])
            result=dict(designer_info=dict(name=source,hash=source,total_images=len(data['analyses']),
                analyzed_images=len(data['analyses']),analysis_date=w.stamp()),
                individual_analyses=data['analyses'],style_analysis=summary,
                individual_analysis_sources=[dict(image_id=f'image_{i+1}',original_image_id=ident)
                    for i,ident in enumerate(data['image_ids'])],
                metadata=dict(analyzer_version='original-garment-nine-v1-20261003',
                              analysis_dimensions=9,aggregation_version=AGGREGATION_VERSION))
            folder=OUT/source
            folder.mkdir(exist_ok=True)
            if not analyzer.save_analysis_result(result,str(folder/'style_analysis.json'),cfg['vlm_model']):
                raise ValueError('Could not persist original summary hierarchy')
            status('running')
        # Freeze all generation requests before viewing any generated output.
        plans=[]
        for plan in protocol['plans']:
            saved=w.load(OUT/plan['source_id']/'train.json')['style_summary']['designer_style_summary']
            prompt=s.render(tasks[plan['task_id']],'saved_nine',dict(profile=saved,summary=''))
            plans.append(dict(plan,prompt=prompt,prompt_sha256=w.text_hash(prompt),
                              saved_summary=w.file_record(OUT/plan['source_id']/'train.json')))
        w.save(OUT/'generation_plan.json',plans)
        evaluator.OUT=OUT
        for plan in plans:
            print('Generating',plan['run_id'],flush=True)
            record=dict(plan,status='started',backend='seed_text_only')
            path=OUT/(plan['run_id']+'.json')
            w.save(path,record)
            response=s.image_call(budget,cfg,plan)
            blob=w.output_bytes(response,cfg)
            with Image.open(io.BytesIO(blob)) as image:
                fmt,dimensions=image.format,image.size
                image.verify()
            dest=OUT/(plan['run_id']+{'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
            dest.write_bytes(blob)
            record.update(status='completed',output=w.file_record(dest),dimensions=dimensions)
            w.save(path,record)
            task=dict(tasks[plan['task_id']],brief=s.BRIEFS[plan['task_id'].rsplit('_',1)[1]])
            evaluator.evaluate(task,plan,dest,[r['path'] for r in task['reference_images']],
                               cfg,prompts,budget)
            status('running')
            print('Scored',len(list(OUT.glob('*.scores.json'))),'/24',flush=True)
        status('completed')
    except Exception as exc:
        status('stopped',error_type=type(exc).__name__,no_automatic_retry=True)
        print('Stopped:',type(exc).__name__,str(exc)[:180],flush=True)
        raise SystemExit(1) from None


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run'])
    parser.add_argument('--execute-api',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':
        prepare()
    elif args.execute_api:
        run()
    else:
        parser.error('Explicit --execute-api required')
