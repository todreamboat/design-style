"""Expanded nine-field comparison. Prepare is offline; paid run requires explicit release.

Literature persona arms are adaptations of Kim & Lim (2026), not full replications.
Flux is a model-family adaptation, not Dress-EM reproduction. Never retry automatically.
"""
import argparse
import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import workflow_benchmark as w
from nine_dimension_pilot import Budget as PilotBudget, render as pilot_render, CONDITIONS as PILOT_CONDITIONS

CONDITIONS = dict(PILOT_CONDITIONS, L_no_persona='KimLim2026_no_persona_adapted',
                  L_multi_expert='KimLim2026_two_stage_multi_expert_adapted')
SOURCES = {
    'persona': 'https://journals.sagepub.com/doi/full/10.1177/15589250261455692',
    'editing_baselines': 'https://arxiv.org/html/2603.22607v1',
}
EXCLUDED_BRANDS = {'1017 ALYX 9SM', 'Acne Studios', 'Akris', '16Arlington', '224 Hommes'}

class StudyBudget(PilotBudget):
    def __init__(self, out, cap):
        if not 0 < cap <= 20:
            raise ValueError('Study budget must be in (0,20] USD')
        self.out, self.cap, self.rows = out, cap, []

def prompt_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 480:
        raise ValueError('Prompt empty or over 480 characters; no truncation/retry')
    return value

def literature_prompt(stage, task, summary, description=''):
    roles = {'none': '', 'designer': '你是一位服装设计师。', 'expert': '你是一位AI图像生成专家。'}
    output = '只输出中文服装设计描述，不超过480字符。' if stage == 'designer' else '只输出可供当前图像模型使用的中文生成描述，不超过480字符；不使用模型专有权重语法。'
    return (roles[stage] + '相同任务要求：' + task['brief'] + '\n必须满足：' + json.dumps(task['hard_constraints'], ensure_ascii=False)
            + '\n相同参考集合的普通总结：' + summary + '\n前阶段描述：' + description
            + '\n当前要求优先；区分内外层，不复制人物、背景或标识；不补造不可观察信息。' + output)

def render(task, condition, profile, summary, reconciled, prompts, literature):
    if condition in ('L_no_persona', 'L_multi_expert'):
        return w.render(task, 'B', profile, literature[condition], prompts)
    return pilot_render(task, condition, profile, summary, reconciled, prompts)

def candidates_from_inventory(inventory, data):
    groups = defaultdict(list)
    with Path(inventory).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['brand'] not in EXCLUDED_BRANDS and row['category'] == 'coats' and row['season_code_provisional']:
                groups[(row['brand'], row['source_group_provisional'])].append(row)
    chosen, brands, hashes = [], set(), set()
    for (brand, prefix), rows in sorted(groups.items()):
        if brand in brands or any(token in prefix.lower().split('_') for token in ('det', 'dtl')):
            continue
        selected = []
        for row in sorted(rows, key=lambda r: r['relative_path']):
            if row['sha256'] not in hashes and row['sha256'] not in {v['sha256'] for v in selected}:
                selected.append(row)
            if len(selected) == 3:
                break
        if len(selected) < 3:
            continue
        entries = []
        for index, row in enumerate(selected):
            path = Path(data).resolve() / row['relative_path']
            if w.digest(path) != row['sha256']:
                raise ValueError('Archive file changed after inventory')
            entries.append(dict(image_id=f'main_b{len(chosen)+1}_r{index+1}', path=str(path), sha256=row['sha256'],
                source='User-authorized local archive; original source URL unavailable', rights_status='licensed_for_research_api',
                garment_group_id=brand+':'+prefix+':'+row['sequence_token'], metadata_status='provisional_not_verified_SKU'))
        chosen.append(dict(brand=brand, reference_images=entries[:2], evaluation_images=entries[2:]))
        brands.add(brand)
        hashes.update(row['sha256'] for row in selected)
        if len(chosen) == 6:
            break
    if len(chosen) != 6:
        raise ValueError('Six unused development brands required')
    return chosen

def prepare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    candidates = candidates_from_inventory(a.inventory, a.data)
    prompts = w.load(w.HERE/'workflow_prompts.json')
    prompts['version'] = 'nine-main-v1-20261006'
    tasks = []
    for i, source in enumerate(candidates):
        for kind in ('appearance', 'structure'):
            if kind == 'appearance':
                brief = '借鉴两张参考图设计外套：外套外表面整体为紫色，具有清晰连续的白色竖条纹；内搭和下装不要求改色。'
                required = ['外套外表面（含领口、肩部、可见衣片）整体为紫色，不保留原色外层衣片；内里及内搭不计',
                            '外套具有白色竖条纹，条纹位于外套而非仅内搭']
                explicit = {'color_tone_usage_characteristics': '外套所有外表面为紫色，白色竖条纹；内搭及下装参考原图',
                            'top_design_characteristics': '参考上装款式，仅将外套表面图案改为白色竖条纹'}
                excluded = ['overall_style_characteristics', 'color_tone_usage_characteristics', 'top_design_characteristics', 'material_selection_characteristics']
            else:
                brief = '借鉴两张参考图设计外套：外套腰部使用一条可见腰带明确收束，外套领口左右对称；其他配色和可见材质参考原图。'
                required = ['外套腰部有一条可见腰带，并形成明确收腰', '外套领口左右结构对称']
                explicit = {'waistline_treatment_characteristics': '外套腰部有可见腰带并明确收腰',
                            'top_design_characteristics': '参考上装款式，领口左右对称',
                            'cut_style_characteristics': '保留参考剪裁中与明确收腰和对称领口不冲突的部分'}
                excluded = ['overall_style_characteristics', 'waistline_treatment_characteristics', 'top_design_characteristics', 'cut_style_characteristics', 'accessory_matching_characteristics']
            tasks.append(dict(source, task_id=f'main_b{i+1}_{kind}', source_id=f'main_b{i+1}', split='development',
                brief=brief, hard_constraints=required, explicit_fields=explicit, preserve=[], style_excluded_dimensions=excluded,
                presentation_checks=['正面展示完整外套，领口、袖部和下摆均可判断', '只输出一个完整设计，不含原参考图缩略图或拼图'],
                artifact_checks=['可见条纹、格纹或装饰线衔接自然，无明确断裂、漂浮或畸变', '外套内外层与开口空间关系自然，无明确错位或融合'],
                eligibility_status='pending_human_source_review', task_origin='researcher_authored_DressED_edit_type_inspired_not_benchmark_replication'))
    cfg = w.load(w.HERE/'workflow_config.json')
    cfg.update(protocol_version='nine-main-v1-20261006', conditions=list(CONDITIONS), repeats=2,
               max_live_tasks=12, max_live_outputs=192, image_seed_base=20261006, schedule_seed=20261006)
    w.validate_tasks(tasks, cfg)
    w.save(out/'config.json', cfg)
    w.save(out/'tasks.json', {'tasks': tasks, 'status': 'expanded_comparison_pending_human_review_not_final_test'})
    w.save(out/'prompts.json', prompts)
    w.save(out/'condition_registry.json', CONDITIONS)
    w.save(out/'literature_sources.json', SOURCES)
    w.save(out/'code_snapshot.json', {name: w.file_record(w.HERE/name) for name in ('nine_main_study.py','nine_dimension_pilot.py','workflow_benchmark.py')})
    w.save(out/'input_manifest.json', {name: w.file_record(out/name) for name in ('config.json','tasks.json','prompts.json','condition_registry.json','literature_sources.json')})
    w.save(out/'review_gate.json', {'human_review_complete': False, 'reviewer': '', 'notes': '',
        'warning': 'No human ground truth. Expanded development allowed only with explicit flag; formal-test claims forbidden.'})
    sheet = Image.new('RGB', (900, 420*len(candidates)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, source in enumerate(candidates):
        for j, ref in enumerate(source['reference_images']+source['evaluation_images']):
            with Image.open(ref['path']) as im:
                sheet.paste(ImageOps.contain(im.convert('RGB'), (290,390)), (300*j,420*i+25))
            draw.text((300*j+5,420*i+5), ref['image_id'], fill='black')
    sheet.save(out/'source_contact_sheet.jpg')
    w.save(out/'flux_candidate_config.json',dict(cfg,image_model='black-forest-labs/flux.2-klein-4b',
        image_provider={'only':['black-forest-labs'],'allow_fallbacks':False},
        image_parameters={'n':1,'aspect_ratio':'2:3'},conditions=['A','C'],authorization_status='pending',
        purpose='DressED_model_family_adaptation_not_DressEM_replication',parameter_support_status='must_verify_public_endpoint_before_execution'))
    w.save(out/'plan.json', {'sources':6,'tasks':12,'conditions':8,'repeats':2,'planned_images':192,
        'maximum_text_requests':456,'planned_flux_images':48,'external_provider_authorized':False,
        'claim_boundary':'Same-backend persona adapters and own ablations; not original paper score reproduction'})
    print('Prepared 192-image comparison; API calls=0', flush=True)

def run(a):
    if not a.execute_api:
        raise ValueError('Explicit paid API execution required')
    kit, out = Path(a.kit), Path(a.out)
    for name, record in w.load(kit/'input_manifest.json').items():
        if w.digest(kit/name) != record['sha256']:
            raise ValueError('Frozen input changed')
    for name, record in w.load(kit/'code_snapshot.json').items():
        if w.digest(w.HERE/name) != record['sha256']:
            raise ValueError('Code changed after freeze')
    review = w.load(kit/'review_gate.json')
    if not review['human_review_complete'] and not a.allow_unreviewed_development:
        raise ValueError('Human source review pending; explicitly release expanded development or finish review')
    cfg, prompts = w.load(kit/'config.json'), w.load(kit/'prompts.json')
    tasks = w.load(kit/'tasks.json')['tasks']
    if cfg['image_model'] != 'bytedance-seed/seedream-5-0-flash' or cfg['image_provider']['only'] != ['seed']:
        raise ValueError('Only already-authorized Seed generation route allowed')
    w.validate_tasks(tasks,cfg)
    out.mkdir(parents=True, exist_ok=False)
    w.load_local_credentials()
    budget = StudyBudget(out,a.budget_usd)
    for name, value in (('frozen_config.json',cfg),('frozen_prompts.json',prompts),('frozen_tasks.json',tasks),('review_gate.json',review)):
        w.save(out/name,value)
    profiles, descriptions, rows = {}, {}, []
    stage = 'extraction_and_literature_prompts'
    try:
        for task in tasks:
            source = task['source_id']
            paths = [r['path'] for r in task['reference_images']]
            if source not in profiles:
                observations = []
                for ref in task['reference_images']:
                    raw = budget.call('text',cfg,prompts['observe'].format(image_id=ref['image_id'],fields=','.join(prompts['fields'])),[ref['path']])
                    observations.append({'image_id':ref['image_id'],'fields':w.fields_check(w.strict_json(raw),prompts['fields'])})
                profile = w.fields_check(w.strict_json(budget.call('text',cfg,prompts['aggregate'].format(fields=','.join(prompts['fields']),observations=json.dumps(observations,ensure_ascii=False)),[])),prompts['fields'])
                if any(v is not None and len(v)>60 for v in profile.values()):
                    raise ValueError('Aggregate too long; no retry')
                summary = prompt_text(budget.call('text',cfg,prompts['free_summary'],paths))
                profiles[source] = dict(profile=profile,summary=summary,observations=observations)
                w.save(out/f'{source}.profile.json',profiles[source])
            summary = profiles[source]['summary']
            reconciled = prompt_text(budget.call('text',cfg,prompts['reconcile_summary'].format(summary=summary,explicit=json.dumps(task['explicit_fields'],ensure_ascii=False)),[]))
            no_persona = prompt_text(budget.call('text',cfg,literature_prompt('none',task,summary),[]))
            designer = prompt_text(budget.call('text',cfg,literature_prompt('designer',task,summary),[]))
            multi = prompt_text(budget.call('text',cfg,literature_prompt('expert',task,summary,designer),[]))
            descriptions[task['task_id']] = dict(reconciled=reconciled,designer_intermediate=designer,literature={'L_no_persona':no_persona,'L_multi_expert':multi})
            w.save(out/f"{task['task_id']}.descriptions.json",descriptions[task['task_id']])
            print('Prepared reference profile and literature prompts:',task['task_id'],flush=True)
        schedule = [(t['task_id'],c,r) for t in tasks for c in CONDITIONS for r in range(cfg['repeats'])]
        random.Random(cfg['schedule_seed']).shuffle(schedule)
        w.save(out/'schedule.json',schedule)
        task_map = {t['task_id']:t for t in tasks}
        stage = 'generation_and_evaluation'
        for task_id,condition,repeat in schedule:
            task=task_map[task_id]
            data=profiles[task['source_id']]
            desc=descriptions[task_id]
            paths=[r['path'] for r in task['reference_images']]
            run_id=f'{task_id}_{condition}_{repeat}'
            prompt=render(task,condition,data['profile'],data['summary'],desc['reconciled'],prompts,desc['literature'])
            row=dict(run_id=run_id,task_id=task_id,source_id=task['source_id'],condition=condition,repeat=repeat,status='started',prompt=prompt,prompt_sha256=w.text_hash(prompt))
            rows.append(row)
            w.save(out/(run_id+'.json'),row)
            image=budget.call('image',cfg,prompt,paths,cfg['image_seed_base']+repeat)
            blob=w.output_bytes(image,cfg)
            with Image.open(io.BytesIO(blob)) as im:
                fmt,dimensions=im.format,im.size
                im.verify()
            dest=out/(run_id+{'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}[fmt])
            dest.write_bytes(blob)
            row.update(status='completed',output=w.file_record(dest),dimensions=dimensions)
            w.save(out/(run_id+'.json'),row)
            judge_cfg=dict(cfg,vlm_model=cfg['judge_model'],vlm_extra={})
            checks=[{'constraint_id':f'{prefix}{i+1}','text':value} for prefix,key in [('t','hard_constraints'),('p','presentation_checks'),('q','artifact_checks')] for i,value in enumerate(task[key])]
            raw=w.strict_json(budget.call('text',judge_cfg,prompts['judge'].format(task=json.dumps({'brief':task['brief'],'constraints':checks},ensure_ascii=False)),[str(dest)]))
            w.judge_checks(raw,[c['constraint_id'] for c in checks])
            metrics={prefix:w.judge_checks({'checks':[r for r in raw['checks'] if r['constraint_id'].startswith(prefix)]},[c['constraint_id'] for c in checks if c['constraint_id'].startswith(prefix)]) for prefix in ('t','p','q')}
            excluded=sorted(set(task['explicit_fields'])|set(task['style_excluded_dimensions']))
            eligible=[f for f in prompts['fields'] if f not in excluded]
            style=w.strict_json(budget.call('text',judge_cfg,prompts['style_judge'].format(excluded_dimensions=json.dumps(excluded),eligible_dimensions=json.dumps(eligible)),paths+[str(dest)]))
            w.save(out/(run_id+'.scores.json'),dict(run_id=run_id,task_id=task_id,source_id=task['source_id'],condition=condition,repeat=repeat,raw_constraints=raw,raw_style=style,task_metrics=metrics['t'],presentation_metrics=metrics['p'],artifact_diagnostics=metrics['q'],style_metrics=w.style_checks(style,eligible),kind='uncalibrated_machine_proxy'))
            w.save(out/'results.json',rows)
            w.save(out/'status.json',dict(status='running',planned=len(schedule),generated=sum(r['status']=='completed' for r in rows),scored=len(list(out.glob('*.scores.json')))))
            print('Completed',len(rows),'/',len(schedule),run_id,flush=True)
        w.save(out/'status.json',dict(status='completed',generated=len(rows),scored=len(rows),formal_test=False,human_review=review['human_review_complete']))
    except Exception as exc:
        if rows and rows[-1]['status']=='started':
            rows[-1].update(status='failed',error_type=type(exc).__name__)
            w.save(out/(rows[-1]['run_id']+'.json'),rows[-1])
        w.save(out/'results.json',rows)
        w.save(out/'status.json',dict(status='stopped',stage=stage,error_type=type(exc).__name__,generated=sum(r['status']=='completed' for r in rows),scored=len(list(out.glob('*.scores.json'))),no_automatic_retry=True))
        raise RuntimeError('Main study stopped; inspect retained records, no automatic retry') from None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','run'])
    p.add_argument('--inventory')
    p.add_argument('--data')
    p.add_argument('--kit')
    p.add_argument('--out',required=True)
    p.add_argument('--budget-usd',type=float,default=20)
    p.add_argument('--execute-api',action='store_true')
    p.add_argument('--allow-unreviewed-development',action='store_true')
    a=p.parse_args()
    (prepare if a.mode=='prepare' else run)(a)

if __name__=='__main__':
    main()
