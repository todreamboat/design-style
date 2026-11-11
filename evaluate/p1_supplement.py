"""Local P1 evidence supplement. No API calls; frozen experiments are read-only."""
import base64
import csv
import html
import json
import mimetypes
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import workflow_benchmark as w
from analyze_completed_comparison import analyze, inference

KIT = w.HERE / 'results/nine_main_kit_v2_20261006'
ROOT = w.HERE / 'results/nine_comparison_run_20261006'
OUT = ROOT / 'p1_supplement_20261012'
SELECTED = ['A', 'B_reconciled', 'C_no_priority', 'L_no_persona']
NAMES = ['Direct reference', 'Reconciled prose', 'Nine fields', 'No persona']


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    result = analyze('seed')  # Includes schedule, completion and output hash validation.
    flux = analyze('flux')
    original = w.load(ROOT / 'comparison_analysis.json')
    for stage, actual in [('seed', result), ('flux', flux)]:
        assert actual['outputs'] == original[stage]['outputs']
        for c, metrics in actual['summary'].items():
            for m, value in metrics.items():
                assert abs(value - original[stage]['summary'][c][m]) < 1e-12
    sources = sorted({r['source'] for r in result['source_means']})
    lookup = {(r['source'], r['condition']): r for r in result['source_means']}
    contrasts = {}
    tests = []
    flat = []
    for c in sorted(result['summary']):
        if c == 'C_no_priority':
            continue
        contrasts[c] = {}
        for m in ('task', 'style'):
            d = [lookup[s, 'C_no_priority'][m] - lookup[s, c][m] for s in sources]
            item = inference(d)
            item['leave_one_source_out'] = {
                s: float(np.mean([v for i, v in enumerate(d) if sources[i] != s]))
                for s in sources}
            contrasts[c][m] = item
            tests.append(item)
    previous = 0.
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda i: tests[i]['exact_sign_flip_p'])):
        previous = max(previous, min(1., (len(tests)-rank)*tests[i]['exact_sign_flip_p']))
        tests[i]['holm_p_main_14_tests'] = previous
    for c, endpoints in contrasts.items():
        for m, item in endpoints.items():
            flat.append(dict(comparator=c, metric=m, delta_pp=100*item['delta'],
                ci_low_pp=100*item['bootstrap_ci95'][0], ci_high_pp=100*item['bootstrap_ci95'][1],
                exact_p=item['exact_sign_flip_p'], holm_p=item['holm_p_main_14_tests'],
                positive_sources=item['positive_sources'], negative_sources=item['negative_sources'],
                loo_min_pp=100*min(item['leave_one_source_out'].values()),
                loo_max_pp=100*max(item['leave_one_source_out'].values())))
    dimension_groups = defaultdict(list)
    sensitivity_groups = defaultdict(list)
    for p in sorted((ROOT / 'seed').glob('*.scores.json')):
        r = w.load(p)
        sensitivity_groups[r['source_id'], r['task_id'], r['condition']].append(r['style_metrics']['known_only_style_proxy'])
        kind = r['task_id'].rsplit('_', 1)[1]
        checks = r['raw_style']['style_checks']
        assert len(checks) == (5 if kind == 'appearance' else 4)
        assert len({x['dimension'] for x in checks}) == len(checks)
        for check in checks:
            assert check['status'] in ('retained', 'partial', 'not_retained', 'unknown')
            dimension_groups[r['condition'], kind, check['dimension']].append(check['status'])
    dims = []
    for (c, kind, dim), labels in sorted(dimension_groups.items()):
        assert len(labels) == 12
        score = sum({'retained': 1, 'partial': .5, 'not_retained': 0, 'unknown': 0}[v] for v in labels)
        known = len(labels) - labels.count('unknown')
        dims.append(dict(condition=c, task_kind=kind, dimension=dim, judgments=len(labels),
            retained=labels.count('retained'), partial=labels.count('partial'),
            not_retained=labels.count('not_retained'), unknown=labels.count('unknown'),
            score_percent=100*score/len(labels), unknown_percent=100*labels.count('unknown')/len(labels),
            known_only_percent=100*score/known if known else None))
    sensitivity = {}
    for c in sorted(result['summary']):
        source_values = [np.mean([np.mean(values) for (source, task, condition),values in sensitivity_groups.items()
            if source == s and condition == c]) for s in sources]
        sensitivity[c] = float(np.mean(source_values))
    payload = dict(analysis_policy=dict(independent_unit='six convenience reference-source blocks',
        aggregation='two generations within task, then two tasks within source',
        inference='post-hoc exploratory two-sided exact sign-flip; symmetric/exchangeable source differences assumed',
        interval='percentile source bootstrap, exhaustive 6^6 draws; small-n descriptive uncertainty',
        multiplicity='separate supplementary family: 14 Seedream main-scheme contrasts, Holm',
        selection='all eight original conditions, all six leave-one-source-out checks, no exclusions',
        human_calibration=False, added_generations=0, original_outputs_validated=240),
        summary=result['summary'], source_means=result['source_means'],
        by_task_kind=result['by_task_kind'], main_contrasts=contrasts, per_dimension=dims,
        known_only_style_sensitivity=sensitivity)
    w.save(OUT / 'analysis.json', payload)
    write_csv(OUT / 'main_contrasts.csv', flat)
    write_csv(OUT / 'source_means.csv', result['source_means'])
    write_csv(OUT / 'per_dimension.csv', dims)
    return payload


def prepare_ablation(tasks, prompts):
    planned = []
    for task in tasks:
        for ref in task['reference_images']:
            if w.digest(ref['path']) != ref['sha256']:
                raise ValueError('Frozen reference image changed: '+ref['image_id'])
        profile = w.load(ROOT / 'seed' / (task['source_id']+'.profile.json'))['profile']
        assert set(profile) == set(prompts['fields'])
        # Reuse the original main wrapper verbatim; only replace its content.
        flat_prompts = dict(prompts)
        flat_prompts['C_no_priority'] = prompts['C_no_priority'].replace('{content}', w.flatten(profile, prompts['fields']))
        flat_prompt = w.render(task, 'C_no_priority', profile, '', flat_prompts)
        base_prompt = w.render(task, 'C_no_priority', profile, '', prompts)
        for repeat in range(2):
            old = w.load(ROOT / 'seed' / f"{task['task_id']}_C_no_priority_{repeat}.json")
            assert old['prompt'] == base_prompt
            planned.append(dict(task_id=task['task_id'], source_id=task['source_id'],
                condition='D_base', repeat=repeat, prompt=flat_prompt,
                prompt_sha256=w.text_hash(flat_prompt), image_seed=20261006+repeat,
                reference_images=task['reference_images'],
                paired_existing_run=old['run_id'], profile_sha256=w.digest(ROOT/'seed'/(task['source_id']+'.profile.json'))))
    w.save(OUT / 'ablation_protocol.json', dict(status='prepared_not_executed', api_calls=0,
        hypothesis='Effect of field labels/JSON representation with identical unedited nine-field values',
        contrast='existing C_no_priority versus new D_base; same common and condition wrapper',
        changes='only representation content: JSON with keys versus ordered flattened values; null becomes unknown',
        interpretation='representation bundle, not isolated labels or token-length effect; not dimensional necessity',
        outcomes=['task conservative adherence', 'unmodified-attribute style proxy'],
        aggregation='source level, identical frozen exclusions and judge',
        config=w.load(KIT/'config.json'), tasks=tasks, prompts=prompts, planned_runs=planned,
        authorization='additional paid dispatch awaits user consent; proposed maximum 5 USD',
        budget_note='client-side returned-usage/reservations do not guarantee provider billing cap'))


def data_uri(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode('ascii')


def blind_page(tasks, prompts):
    labels = dict(zip(prompts['fields'], ['整体风格', '搭配', '上装', '下装', '腰线', '配饰', '配色', '材质外观', '剪裁']))
    cases = [(task, c) for task in tasks for c in SELECTED]
    random.Random(20261012).shuffle(cases)
    mapping, cards = [], []
    selections = [('未填写', ''), ('满足', 'met'), ('不满足', 'violated'), ('无法判断', 'unknown')]
    style_options = [('未填写', ''), ('保留', 'retained'), ('部分保留', 'partial'), ('未保留', 'not_retained'), ('无法判断', 'unknown')]
    for i, (task, c) in enumerate(cases, 1):
        case_id = f'case_{i:03d}'
        run_id = f"{task['task_id']}_{c}_0"
        record = w.load(ROOT/'seed'/(run_id+'.json'))
        eligible = [x['dimension'] for x in w.load(ROOT/'seed'/(run_id+'.scores.json'))['raw_style']['style_checks']]
        mapping.append(dict(case_id=case_id, run_id=run_id, condition=c, task_id=task['task_id'],
            output_sha256=record['output']['sha256'], eligible_dimensions=eligible))
        pictures = ''.join('<figure><img src="'+data_uri(p)+'"><figcaption>'+caption+'</figcaption></figure>'
            for p, caption in [(r['path'], '参考图'+str(j)) for j,r in enumerate(task['reference_images'],1)]
                + [(record['output']['path'], '待评价生成图')])
        def selector(key, caption, options):
            return '<label>'+html.escape(caption)+' <select data-case="'+case_id+'" data-key="'+key+'">'+''.join(
                '<option value="'+v+'">'+n+'</option>' for n,v in options)+'</select></label>'
        fields = ''.join(selector('t'+str(j), constraint, selections) for j,constraint in enumerate(task['hard_constraints'],1))
        fields += ''.join(selector('p'+str(j), constraint, selections) for j,constraint in enumerate(task['presentation_checks'],1))
        fields += ''.join(selector('q'+str(j), constraint, selections) for j,constraint in enumerate(task['artifact_checks'],1))
        fields += ''.join(selector(dim, labels[dim], style_options) for dim in eligible)
        cards.append('<section><h2>'+case_id+'</h2><p>'+html.escape(task['brief'])+'</p><div class="images">'+pictures+
            '</div>'+fields+'<label>可见证据或疑问<textarea data-case="'+case_id+'" data-key="evidence"></textarea></label></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>服装属性盲评</title>
<style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:24px}section{border-top:2px solid #ddd;padding:24px 0}.images{display:flex;gap:16px}figure{flex:1;margin:0}img{width:100%;height:420px;object-fit:contain}label{display:block;margin:12px 0}textarea{display:block;width:90%;height:70px}select{margin-left:12px}header{position:sticky;top:0;background:white;padding:10px;border-bottom:1px solid #ddd}button{padding:10px}</style>
<header>评价人：<input id="reviewer"> <button onclick="exportLabels()">下载评价JSON</button><span id="count"></span></header>
<h1>服装属性核验：48张</h1><p>按随机顺序评价。不要打开映射文件或机器分数。任务满足只看生成图；风格保留比较参考图与生成图，仅评价列出的未要求改变属性。主要特征或参考图中明确存在的一种变化计保留；明显变化但仍保留部分特征计部分保留；不可见选无法判断。参考互有差异时，不要求同时复制互斥属性。请注明可见证据。</p>
<p>本页不联网，填写内容暂存于当前页面，刷新或关闭会丢失。按评价者确认：未填写的检查项按满足、风格项按保留导出；已填写值原样保留，默认补全项另行记录。请在关闭前保存JSON。此核验不是设计师满意度研究，也不自动改变旧评分。</p>'''+''.join(cards)+'''
<script>const elements=[...document.querySelectorAll('[data-case]')];
function exportLabels(){const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('请填写评价人');return}const cases={},defaulted=[];elements.forEach(e=>{let value=e.value;if(e.tagName==='SELECT'&&!value){value=/^[tpq]\d+$/.test(e.dataset.key)?'met':'retained';defaulted.push({case_id:e.dataset.case,key:e.dataset.key,value});}(cases[e.dataset.case]??={})[e.dataset.key]=value});const blob=new Blob([JSON.stringify({protocol:'p1-blind-48-v1',reviewer,completed_at:new Date().toISOString(),blank_policy:'user_confirmed_reviewed_blank_means_met_or_retained',defaulted,cases},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
elements.forEach(e=>e.addEventListener('change',()=>document.getElementById('count').textContent=elements.filter(e=>e.tagName==='SELECT'&&e.value).length+'项已填写'));</script></html>'''
    (OUT/'blind_review.html').write_text(page, encoding='utf-8')
    w.save(OUT/'blind_mapping_DO_NOT_OPEN_BEFORE_RATING.json', dict(status='labels_pending',
        selection='repeat 0, all 12 tasks, four matched conditions; balanced convenience reliability audit',
        selection_seed=20261012, created_before_human_labels=True, cases=mapping,
        planned_analysis='task exact agreement and Cohen kappa; ordinal style weighted kappa excluding unknown plus unknown agreement; clustering by six sources; no general human-preference inference'))


def figures(tasks, result):
    os.environ.setdefault('MPLCONFIGDIR', str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family':'Arial', 'font.size':7, 'svg.fonttype':'none', 'pdf.fonttype':42})
    def export(fig, name):
        fig.savefig(OUT/(name+'.svg'), facecolor='white')
        fig.savefig(OUT/(name+'.pdf'), facecolor='white')
        fig.savefig(OUT/(name+'.png'), dpi=600, facecolor='white')
        plt.close(fig)
    width_inches = 180 / 25.4
    fig, axes = plt.subplots(1,2, figsize=(width_inches,80/25.4), layout='constrained')
    compared = ['A','B','B_reconciled','L_no_persona','L_multi_expert','C','D']
    short = ['Direct reference','Free prose','Reconciled prose','No persona','Two roles','Edited fields','Edited flat']
    for ax, metric, title in zip(axes, ['task','style'], ['a  Task constraints','b  Unmodified attributes']):
        for i,c in enumerate(compared):
            item = result['main_contrasts'][c][metric]
            vals = 100*np.asarray(item['source_differences'])
            mean = 100*item['delta']
            low,high = 100*np.asarray(item['bootstrap_ci95'])
            ax.scatter(vals, np.full(6,i)+np.linspace(-.12,.12,6), s=9, c='#999999', zorder=2)
            ax.plot([low,high],[i,i], color='#0072B2', lw=1.4)
            ax.scatter([mean],[i], s=18, marker='D', c='#0072B2',zorder=3)
        ax.axvline(0,color='black',lw=.6,ls='--')
        ax.set_yticks(range(7),short)
        ax.invert_yaxis()
        ax.set_xlabel('Nine fields minus comparator (percentage points)')
        ax.set_title(title,loc='left',fontweight='bold',fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    export(fig,'main_uncertainty')
    image_manifest = []
    for kind in ['appearance','structure']:
        chosen = [t for t in tasks if t['task_id'].endswith(kind)]
        # Two sources per plate: six plates retain all sources, no score-based choice.
        for batch in range(3):
            fig,axes=plt.subplots(2,6,figsize=(width_inches,115/25.4),layout='constrained')
            for row,t in enumerate(chosen[batch*2:batch*2+2]):
                items=[(r['path'],'Reference '+str(i)) for i,r in enumerate(t['reference_images'],1)]
                for c,name in zip(SELECTED,NAMES):
                    record=w.load(ROOT/'seed'/f"{t['task_id']}_{c}_0.json")
                    items.append((record['output']['path'],name))
                for col,(path,caption) in enumerate(items):
                    with Image.open(path) as im:
                        axes[row,col].imshow(im.copy())
                    axes[row,col].axis('off')
                    axes[row,col].set_title(caption,fontsize=6)
                    if col==0:
                        axes[row,col].text(0,-.04,t['source_id'],transform=axes[row,col].transAxes,fontsize=7)
                    image_manifest.append(dict(plate=f'{kind}_examples_{batch+1}',source=t['source_id'],
                        task=t['task_id'],caption=caption,path=path,sha256=w.digest(path),
                        processing='complete image, proportional display; no crop, retouch, color adjustment'))
            export(fig, f'{kind}_examples_{batch+1}')
    w.save(OUT/'figure_manifest.json',dict(backend='python/matplotlib',
        selection='all six sources, both task kinds, repeat 0, four fixed conditions, not chosen by scores',
        quantitative='six paired source differences; diamond mean; exhaustive percentile source-bootstrap interval',
        images=image_manifest, source_data=['source_means.csv','main_contrasts.csv'],
        target='generic 180 mm width; final venue specifications not yet checked'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = w.load(KIT/'tasks.json')['tasks']
    prompts = w.load(KIT/'prompts.json')
    result = summarize()
    prepare_ablation(tasks,prompts)
    blind_page(tasks,prompts)
    figures(tasks,result)
    print(json.dumps(dict(outputs_validated=240, analysis=str(OUT/'analysis.json'),
        blind_cases=48, new_generation_status='prepared_not_executed', api_calls=0,
        key_contrasts={c:result['main_contrasts'][c] for c in ['A','B_reconciled','L_no_persona']}),ensure_ascii=False))


if __name__ == '__main__':
    main()
