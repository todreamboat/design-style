"""Offline closeout of the user-accepted partial batch. Never calls an API."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
from urllib.parse import quote
from PIL import Image
import workflow_benchmark as w

ARMS = ('task_only', 'saved_summary', 'saved_nine')
LABELS = {'task_only': '仅任务', 'saved_summary': '保存普通摘要', 'saved_nine': '保存九维描述'}


def summarize(rows):
    result = {}
    for arm in ARMS:
        selected = [r for r in rows if r['condition'] == arm]
        sources = defaultdict(lambda: defaultdict(list))
        for row in selected:
            sources[row['source_id']][row['task_id']].append(row)
        values = {}
        for name, key in [('task', 'task_metrics'), ('presentation', 'presentation_metrics'),
                          ('artifact', 'artifact_metrics')]:
            brand_values = [statistics.mean(statistics.mean(x[key]['conservative_adherence'] for x in group)
                            for group in tasks.values()) for tasks in sources.values()]
            values[name+'_brand_balanced_percent'] = 100*statistics.mean(brand_values)
            met = sum(r[key]['met_count'] for r in selected)
            total = sum(r[key]['constraint_count'] for r in selected)
            values[name+'_met'] = met
            values[name+'_checks'] = total
            values[name+'_pooled_percent'] = 100*met/total
        result[arm] = dict(images=len(selected), brands=len(sources), **values)
    return result


def main(out, report):
    protocol = w.load(out/'protocol.json')
    tasks = {t['task_id']: t for t in protocol['tasks']}
    schedule = [(r['task_id'], r['condition'], r['repeat']) for r in protocol['schedule']]
    expected = set(schedule)
    records = {}
    scores = {}
    for path in sorted((out/'outputs').glob('*.json')):
        row = w.load(path)
        key = (row['task_id'], row['condition'], row['repeat'])
        if key in records or key not in expected:
            raise ValueError('Duplicate or unplanned generation')
        if row['generation_image_inputs']:
            raise ValueError('Historical images unexpectedly supplied to generator')
        if w.digest(row['output']['path']) != row['output']['sha256']:
            raise ValueError('Output hash mismatch')
        with Image.open(row['output']['path']) as im:
            im.verify()
        records[key] = row
    for path in sorted((out/'scores').glob('*.json')):
        row = w.load(path)
        key = (row['task_id'], row['condition'], row['repeat'])
        if key in scores or key not in records:
            raise ValueError('Duplicate score or missing corresponding image')
        task = tasks[row['task_id']]
        if row['source_id'] != task['source_id']:
            raise ValueError('Source mapping mismatch')
        run_id = f'{key[0]}_{key[1]}_{key[2]}'
        raw = w.load(out/'judgments'/(run_id+'.constraints.json'))['value']
        for prefix, metric in [('t', 'task_metrics'), ('p', 'presentation_metrics'), ('q', 'artifact_metrics')]:
            measured = w.judge_checks({'checks': [x for x in raw['checks'] if x['constraint_id'].startswith(prefix)]},
                                     [prefix+'1', prefix+'2'])
            if measured != row[metric]:
                raise ValueError('Stored metric differs from raw judgments')
        scores[key] = row
    rows = list(scores.values())
    if len(records) != 128 or len(rows) != 128:
        raise ValueError('Closeout requires the user-accepted 128-image batch; do not silently change scope')
    blocks = defaultdict(dict)
    for key, row in scores.items():
        blocks[(key[0], key[2])][key[1]] = row
    triples = {k: v for k, v in blocks.items() if set(v) == set(ARMS)}
    paired_rows = [row for group in triples.values() for row in group.values()]
    budget = w.load(out/'budget_ledger.json')
    by_signature = defaultdict(list)
    for index, request in enumerate(budget['requests'], 1):
        base = request.get('retry_of', request['request_sha256'].split(':')[0])
        by_signature[base].append((index, request))
    style_audit = []
    constraints_audit = []
    for key in scores:
        run_id = f'{key[0]}_{key[1]}_{key[2]}'
        for suffix, destination in [('style', style_audit), ('constraints', constraints_audit)]:
            cached = w.load(out/'judgments'/(run_id+'.'+suffix+'.json'))
            matched = []
            for index, request in by_signature[cached['request_sha256']]:
                path = out/'calls'/f'request_{index:04d}'/'call_00001.text.json'
                if not path.exists():
                    continue
                raw = w.load(path)
                try:
                    same = w.strict_json(raw['response']) == cached['value']
                except ValueError:
                    same = False
                if same:
                    matched.append((index, raw))
            if not matched:
                raise ValueError('Cannot trace judgment to original API response')
            index, raw = matched[-1]
            paths = raw['images']
            actual_output_is_last = bool(paths) and paths[-1]['sha256'] == records[key]['output']['sha256']
            if suffix == 'style':
                wrong = raw['prompt'].startswith('图1和图2为服装风格参考，图3为生成结果。') and len(paths) == 5
                destination.append({'run_id':run_id, 'api_request_index': index,
                    'image_count': len(paths), 'actual_output_is_last': actual_output_is_last,
                    'role_index_mismatch': wrong, 'usable_for_method_comparison': not wrong,
                    'prompt_prefix':raw['prompt'][:40]})
            else:
                if len(paths) != 1 or not actual_output_is_last:
                    raise ValueError('Task evaluation was not mapped to the actual generated image')
                destination.append({'run_id':run_id,'api_request_index':index,'correct_output_mapping':True})
    if sum(x['role_index_mismatch'] for x in style_audit) != 128:
        raise ValueError('Unexpected style audit scope; inspect rather than reuse this report')
    source_counts = []
    for source in protocol['sources']:
        sid = source['source_id']
        counts = Counter(r['condition'] for r in rows if r['source_id'] == sid)
        source_counts.append({'source_id':sid,'brand':source['canonical_key'],
            **{arm:counts[arm] for arm in ARMS},
            'complete_triple_blocks':sum(tasks[k[0]]['source_id']==sid for k in triples)})
    returned = budget['prior_probe_usd']+sum(r.get('returned_cost_usd',0) for r in budget['requests'])
    accounted = budget['prior_probe_usd']+sum(r['accounted_usd'] for r in budget['requests'])
    result = {'status':'user_finalized_partial_batch_with_invalid_style_endpoint',
        'closed_date':'2026-10-22','planned_images':len(expected),'generated_images':len(records),
        'scored_images':len(rows),'missing_images':len(expected-set(records)),
        'independent_brands':len(protocol['sources']),
        'unique_generated_image_hashes':len({r['output']['sha256'] for r in records.values()}),
        'all_available':summarize(rows),'complete_triples':summarize(paired_rows),
        'complete_triple_blocks':len(triples), 'complete_triple_images':len(paired_rows),
        'source_counts':source_counts,'task_type_counts':dict(Counter(tasks[r['task_id']]['task_type'] for r in rows)),
        'style_endpoint':{'valid':False,'affected_outputs':128,'reason':'Four reference images plus generated image 5 were sent, but the prompt identified image 3 as generated. Raw evidence in sampled responses describes historical image 3, not the output.',
                          'action':'Quarantine all style scores. Do not infer rankings or test style superiority.'},
        'style_mapping_audit':style_audit,'task_mapping_audit':constraints_audit,
        'financials':{'returned_cost_usd':returned,'charged_or_reserved_usd':accounted,
                      'cap_usd':25,'note':'Returned API usage is not a final account invoice; unknown failed charges retain reservations.'},
        'human_validation':'not performed for this batch; historical 48-image audit does not apply',
        'scope':'brand-proxy archive reuse on researcher-authored tasks; no verified cross-series split or parameter training',
        'selection_note':'Fixed shuffled execution stopped on HTTP 402. Pairing is defined by task and repeat, not scores. Balanced-subset analysis is a post-interruption exploratory analysis.',
        'provenance':{'protocol_sha256':w.digest(out/'protocol.json'),'analyzer_sha256':w.digest(__file__)}}
    w.save(out/'analysis_partial.json',result)
    w.save(out/'closeout.json',{'date':'2026-10-22','user_decision':'accept existing data; no further paid calls or missing-image generation',
                             'images':128,'analysis':'analysis_partial.json','style_endpoint_valid':False})
    with (out/'results_partial.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['run_id','source_id','task_type','condition','repeat','complete_triple',
                'task_percent','presentation_percent','artifact_percent','style_valid',
                'quarantined_style_proxy_do_not_use','image_path','image_sha256']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for key,row in sorted(scores.items()):
            writer.writerow({'run_id':f'{key[0]}_{key[1]}_{key[2]}','source_id':row['source_id'],
                'task_type':tasks[key[0]]['task_type'],'condition':key[1],'repeat':key[2],
                'complete_triple':(key[0],key[2]) in triples,
                'task_percent':100*row['task_metrics']['conservative_adherence'],
                'presentation_percent':100*row['presentation_metrics']['conservative_adherence'],
                'artifact_percent':100*row['artifact_metrics']['conservative_adherence'],
                'style_valid':False,'quarantined_style_proxy_do_not_use':row['style_metrics']['style_retention_proxy'],
                'image_path':records[key]['output']['path'],'image_sha256':records[key]['output']['sha256']})
    write_report(report,result,out)
    write_gallery(out,protocol,records,scores,triples)
    print(json.dumps({'images':128,'complete_triple_blocks':len(triples),'all_available':result['all_available'],
                      'style_endpoint_valid':False,'returned_cost':returned},ensure_ascii=False,indent=2))


def write_report(path, result, out):
    lines=['# 补充实验结果整理（128张，2026-10-22）','',
        '## 结论','',
        '已收集并完成机器评价的生成图共128张，覆盖10个品牌、30项任务。用户决定以现有数据结项，不再补生成原计划中缺失的52张。',
        '任务符合、展示和局部缺陷检查可追溯到对应生成图；风格评价存在输入编号错位，全部128份风格分数隔离，不用于方法排名、显著性检验或论文结论。',
        '此前声称评价图片编号已修正，但该修正未落实到实际执行协议，属于执行核对疏漏。这里保留错误原始记录，不将它解释为九维方法效果较差。','',
        '## 1. 已完成规模与覆盖','',
        '| 条件 | 已生成且评分 | 原计划 | 未完成 | 核心任务判断数 |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        s=result['all_available'][arm]
        lines.append(f'| {LABELS[arm]} | {s["images"]} | 60 | {60-s["images"]} | {s["task_checks"]} |')
    lines.extend(['','每品牌4张历史作品，逐图提取九维、聚合并保存，后续生成仅使用保存文本和任务。两种保存形式使用相同观察信息；任务结合模型相同。品牌是风格代理单元，不是已识别的单一个人设计师。系列文件名仍属候选信息，本批不宣称已核验跨系列划分。','',
        '## 2. 可用的机器评价','',
        '下表先按重复、任务、品牌汇总，再对品牌等权平均。三组覆盖并不完全相同，因此全部可用结果只作描述性统计。指标不是人工满意度，也不能把自动检查通过率等同于真实图像质量。','',
        '| 条件 | 图数 | 任务符合率（%） | 展示检查符合率（%） | 局部缺陷检查通过率（%） | 风格保留 |',
        '|---|---:|---:|---:|---:|---|'])
    for arm in ARMS:
        s=result['all_available'][arm]
        lines.append(f'| {LABELS[arm]} | {s["images"]} | {s["task_brand_balanced_percent"]:.2f} | {s["presentation_brand_balanced_percent"]:.2f} | {s["artifact_brand_balanced_percent"]:.2f} | 无效，需重评 |')
    lines.extend(['','任务指标处于天花板，不能区分三种方法，也不能据此认定九维优于基线或三组等效。新批次尚未进行人工核验，旧批次48张人工评价不能移用到这些图像上。','',
        '## 3. 同任务、同重复的完整配对子集','',
        f'三条件均有结果的单元共{result["complete_triple_blocks"]}个，包含{result["complete_triple_images"]}张图，每条件24张，覆盖10个品牌。单元选择只看是否齐全，不看评分。先在相同任务及重复编号内配对，再按任务与品牌汇总。','',
        '| 条件 | 图数 | 任务符合率（%） | 风格保留 |','|---|---:|---:|---|'])
    for arm in ARMS:
        s=result['complete_triples'][arm]
        lines.append(f'| {LABELS[arm]} | {s["images"]} | {s["task_brand_balanced_percent"]:.2f} | 无效，需重评 |')
    lines.extend(['','这个子集可以在相同输入任务范围内对照现有任务指标，但不能恢复错误的风格评分。它是中断后的探索性分析，不是原180张完整协议的替代性“预设主分析”。未完成项不填为0分，也不伪造有效输出。','',
        '## 4. 风格评分为什么不能用','',
        '- 128份实际调用均发送4张历史参考图和最后一张生成图，共5张图片。',
        '- 实际调用提示仍为“图1和图2为服装风格参考，图3为生成结果”，而不是把第5张指定为生成图。',
        '- 例如v01_color_saved_nine_1的真实输出是深蓝白领、系腰带、粉色高领内搭；机器风格证据却描述驼色大衣、紫色内搭、无腰带，正对应第3张历史参考作品。该记录位于calls/request_0450/call_00001.text.json。',
        '- 输入编号错误影响评价对象，不是汇总公式的小误差，不能通过重新求均值修复。',
        '- 因此不报告这些分数的条件均值或p值，不把它们解释成“九维与摘要差不多”或“九维落后”。','',
        '修复只需要对现有128张输出重新做风格评价，不需要再生图。可使用明确标注图序的评价请求或人工核验；需要外部API时，先另行确认账户余额及剩余预算。本次整理没有新增任何API调用。','',
        '## 5. 品牌覆盖','',
        '| 品牌索引 | 来源 | 仅任务 | 普通摘要 | 九维 | 完整三条件配对单元 |','|---|---|---:|---:|---:|---:|'])
    for s in result['source_counts']:
        lines.append(f'| {s["source_id"]} | {s["brand"]} | {s["task_only"]} | {s["saved_summary"]} | {s["saved_nine"]} | {s["complete_triple_blocks"]} |')
    lines.extend(['','## 6. 对论文的可用结论','',
        '可支持的陈述：保存文本驱动的工作流已在10个品牌来源的30项任务中产出128张新图；生成端未再次输入历史原图，三条件的已完成输出均接受相同的机器任务检查。',
        '不能支持的陈述：九维在本批中保留风格更好、显著优于摘要、已获人工验证、测量真实设计师偏好或证实跨系列泛化。不能把旧批次风格分数直接拼到本批128张上。',
        '论文中的方法优势须等待现有图像的有效风格重评；若暂不重评，本批只能作为工作流覆盖和任务可执行性的补充，不应作为风格优势主表。','',
        '## 7. 文件与费用','',
        f'- 输出目录：`{out.as_posix()}`。',
        '- `results_partial.csv`：128张逐图结果，风格字段明确标记为隔离值。',
        '- `analysis_partial.json`：覆盖、配对子集、图片映射核验和费用。',
        '- `results_gallery.html`：按任务和重复并排查看三条件；未完成位置明确留空。',
        '- `closeout.json`：用户接受现有数据结项，不再补生成的决定。',
        f'- API已返回usage合计约{result["financials"]["returned_cost_usd"]:.4f}美元；含失败请求保守预留约{result["financials"]["charged_or_reserved_usd"]:.4f}美元；这不是账户最终账单。',
        '- 未覆盖或修改原图、原始响应、旧评分、旧协议或旧论文。',''])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines),encoding='utf-8')


def write_gallery(out, protocol, records, scores, triples):
    lines=['<!doctype html><meta charset="utf-8"><title>128张补充实验对照</title>',
        '<style>body{font:15px sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%;height:390px;object-fit:contain}.cell{border:1px solid #ddd;padding:10px}section{margin-bottom:30px}.warning{background:#fff0ce;padding:16px}.missing{color:#777}</style>',
        '<h1>128张补充实验对照</h1><p class="warning">风格自动评分存在参考图／生成图编号错位，全部隔离，不在此展示排名。以下保留全部有效输出；空格代表未完成，不代表低分。任务评分为机器检查，非人工结论。</p>']
    for task in protocol['tasks']:
        for repeat in range(2):
            lines.append('<section><h2>'+html.escape(task['task_id'])+f' · 重复{repeat}</h2><p>'+html.escape(task['brief'])+'</p>')
            lines.append('<p>'+('三条件齐全' if (task['task_id'],repeat) in triples else '三条件未齐全')+'</p><div class="grid">')
            for arm in ARMS:
                key=(task['task_id'],arm,repeat)
                lines.append('<div class="cell"><h3>'+LABELS[arm]+'</h3>')
                if key in records:
                    image=Path(records[key]['output']['path']);url=quote('outputs/'+image.name)
                    lines.append(f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a>')
                    row=scores[key]
                    lines.append(f'<p>核心任务：{row["task_metrics"]["met_count"]}/{row["task_metrics"]["constraint_count"]}；风格评分：无效，待重评。</p>')
                else:
                    lines.append('<p class="missing">未完成，无生成图或评分；未补值。</p>')
                lines.append('</div>')
            lines.append('</div></section>')
    (out/'results_gallery.html').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--report',required=True)
    a=p.parse_args()
    main(Path(a.out).resolve(),Path(a.report).resolve())
