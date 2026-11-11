"""User-authorized durable retries; fixed inputs, no score-dependent decisions."""
import base64
import json
import math
from pathlib import Path
import time
import workflow_benchmark as w


class DurableClient(w.Client):
    def request(self, kind, payload):
        data = super().request(kind, payload)
        if kind == 'image':
            w.save(self.out / 'image_response.pending.json', data)
        else:
            message = data.get('choices', [{}])[0]
            # Retain partial final text on truncation, but do not log reasoning text.
            content = message.get('message', {}).get('content')
            w.save(self.out / 'call_00001.text.json', {
                'prompt': payload['messages'][0]['content'][0]['text'],
                'response': content if isinstance(content, str) else '',
                'images': getattr(self, 'input_records', []),
                'finish_reason': message.get('finish_reason')})
        return data


class AutoBudget:
    automatic_retry = True

    def __init__(self, base, sleep=time.sleep):
        self.base = base
        self.out, self.path, self.data = base.out, base.path, base.data
        self.sleep = sleep

    def spent(self):
        return self.base.spent()

    def rows(self, signature):
        return [(i, r) for i, r in enumerate(self.data['requests'], 1)
                if r['request_sha256'] == signature or r.get('retry_of') == signature
                or r['request_sha256'].startswith(signature + ':')]

    def folder(self, index):
        return self.out / 'calls' / f'request_{index:04d}'

    def classify(self, index, row):
        folder = self.folder(index)
        error_file, meta_file = folder/'call_00001.error.json', folder/'call_00001.metadata.json'
        error = w.load(error_file) if error_file.exists() else {}
        meta = w.load(meta_file) if meta_file.exists() else {}
        http = error.get('http_status', meta.get('http_status'))
        if http is not None and http != 200:
            return 'transient_http' if http in (408, 409, 425, 429) or http >= 500 else 'permanent_http'
        chain = error.get('exception_chain_types', [])
        if 'PermissionError' in chain and not getattr(self, 'authorized_network_permission_recovery', False):
            return 'permanent_http'
        if any(x in chain for x in ('SSLCertVerificationError', 'CertificateError')):
            return 'certificate_failure'
        if error.get('error') == 'network_error':
            return 'network_failure'
        if row.get('state') == 'invalid_response':
            return 'invalid_response'
        text_file = folder/'call_00001.text.json'
        text = w.load(text_file) if text_file.exists() else {}
        if text.get('finish_reason') == 'length':
            return 'truncation'
        usage = meta.get('usage') or {}
        if (http == 200 and not text_file.exists() and row.get('kind') == 'text'
                and usage.get('completion_tokens', 0) >= row.get('effective_max_tokens', 8192)):
            return 'truncation'
        if row.get('state') == 'returned':
            return 'returned'
        if http == 200:
            return 'invalid_response'
        return 'ambiguous_dispatch'

    def wait(self, signature, failure_count, reason):
        latest = self.rows(signature)
        if latest and self.classify(*latest[-1]) == 'permanent_http':
            raise RuntimeError('Non-retryable local network permission failure')
        seconds = min(300, 15 * 2**min(max(0, failure_count-1), 5))
        status_file = self.out/'status.json'
        status = w.load(status_file) if status_file.exists() else {}
        status.update(state='retry_wait', retry_reason=reason,
                      retry_delay_seconds=seconds, retry_request_sha256=signature,
                      accounted_usd=self.spent(), updated_utc=w.stamp())
        w.save(status_file, status)
        print(f'Retry wait {seconds}s: {reason}; accounted ${self.spent():.4f}', flush=True)
        while seconds:
            chunk = min(30, seconds)
            self.sleep(chunk)
            seconds -= chunk
            status.update(updated_utc=w.stamp(), retry_remaining_seconds=seconds)
            w.save(status_file, status)

    def recover(self, index, row, kind, cfg, prompt, paths):
        folder = self.folder(index)
        if kind == 'text':
            raw = w.load(folder/'call_00001.text.json')
            if raw['prompt'] != prompt or raw['images'] != [w.file_record(p) for p in paths]:
                raise ValueError('Completed request payload mismatch; stop, do not retry')
            return raw['response']
        pending = folder/'image_response.pending.json'
        if pending.exists():
            return w.load(pending)
        if row.get('validated_output'):
            record = row['validated_output']
            if w.digest(record['path']) != record['sha256']:
                raise ValueError('Completed image changed; stop, do not retry')
            return {'data': [{'b64_json': base64.b64encode(Path(record['path']).read_bytes()).decode('ascii')}]}
        raise ValueError('Completed image response unavailable; inspect before redispatch')

    def call(self, kind, cfg, prompt, paths, seed, signature):
        if kind == 'image' and paths:
            raise ValueError('Historical images forbidden at generation endpoint')
        if len(prompt) > 100000 or len(paths) > 5:
            raise ValueError('Request exceeds fixed input envelope')
        while True:
            history = self.rows(signature)
            request_cfg = dict(cfg)
            prior_max = cfg.get('text_max_tokens', 8192)
            if history:
                index, row = history[-1]
                reason = self.classify(index, row)
                if reason == 'returned':
                    return self.recover(index, row, kind, cfg, prompt, paths)
                if reason in ('permanent_http', 'certificate_failure'):
                    raise RuntimeError(f'Non-retryable {reason}; original call {index}')
                for j, previous in history:
                    if self.classify(j, previous) == 'truncation':
                        prior_max = max(prior_max, min(32768, previous.get('effective_max_tokens', 8192)*2))
                next_reserve = .10 if kind == 'image' else (.50 if prior_max > 16384 else .35)
                if self.spent() + next_reserve > min(25, self.base.phase_cap):
                    raise ValueError('Authorized budget ceiling reached before retry/dispatch')
                self.wait(signature, len(history), reason)
            request_cfg['text_max_tokens'] = prior_max
            if prior_max > 8192:
                request_cfg['timeout_seconds'] = max(360, request_cfg['timeout_seconds'])
            reserve = .10 if kind == 'image' else (.50 if prior_max > 16384 else .35)
            if self.spent() + reserve > min(25, self.base.phase_cap):
                raise ValueError('Authorized budget ceiling reached before retry/dispatch')
            physical = signature if not history else signature + f':auto_retry_{len(history)}'
            row = {'request_sha256': physical, 'retry_of': signature, 'kind': kind,
                   'state': 'dispatched', 'accounted_usd': reserve, 'reserve_usd': reserve,
                   'effective_max_tokens': prior_max, 'retry_number': len(history),
                   'prompt_sha256': w.text_hash(prompt), 'seed': seed,
                   'model': cfg['image_model'] if kind == 'image' else cfg['vlm_model'],
                   'authorization': 'user_auto_retry_until_complete_within_25usd_20261021'}
            self.data['requests'].append(row)
            index = len(self.data['requests'])
            folder = self.folder(index)
            w.save(self.path, self.data)
            status_file = self.out/'status.json'
            status = w.load(status_file) if status_file.exists() else {}
            status.update(state='running', current_call=index, retry_number=len(history),
                          accounted_usd=self.spent(), updated_utc=w.stamp())
            w.save(status_file, status)
            client = DurableClient(request_cfg, folder)
            client.input_records = [w.file_record(p) for p in paths]
            failure = None
            try:
                result = client.image(prompt, paths, seed) if kind == 'image' else client.text(prompt, paths)
                row['state'] = 'returned'
            except (RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
                # Never catch budget, input-envelope or local-file failures here.
                row.update(state='failed', failure_type=type(exc).__name__, failure=str(exc)[:200])
                failure = exc
            finally:
                meta_file = folder/'call_00001.metadata.json'
                if meta_file.exists():
                    cost = (w.load(meta_file).get('usage') or {}).get('cost')
                    if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0:
                        row.update(returned_cost_usd=cost, accounted_usd=cost)
                w.save(self.path, self.data)
            if failure is None:
                return result
            reason = self.classify(index, row)
            if reason in ('permanent_http', 'certificate_failure'):
                raise RuntimeError(f'Non-retryable {reason}; inspect sanitized call {index}') from failure

    def reject(self, signature, error):
        history = self.rows(signature)
        if not history:
            raise ValueError('Cannot reject a request that was not dispatched')
        index, row = history[-1]
        row.update(state='invalid_response', validation_error=str(error)[:250])
        w.save(self.path, self.data)

    def consume_image(self, signature, path):
        index, row = self.rows(signature)[-1]
        row['validated_output'] = w.file_record(path)
        w.save(self.path, self.data)
        (self.folder(index)/'image_response.pending.json').unlink(missing_ok=True)


def validated_cache(path, budget, kind, cfg, prompt, paths, validate, seed=None, plan=False):
    signature = w.text_hash(json.dumps([kind, cfg, prompt,
        [w.file_record(p) for p in paths], seed], sort_keys=True, ensure_ascii=False))
    if path.exists():
        saved = w.load(path)
        if saved['request_sha256'] != signature:
            raise ValueError('Cached request changed; cannot retry a changed experiment')
        return validate(saved['value'])
    while True:
        result = budget.call(kind, cfg, prompt, paths, seed, signature)
        try:
            value = validate(result)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            budget.reject(signature, exc)
            continue
        record = {'request_sha256': signature, 'value': value}
        if plan:
            original = w.strict_json(result) if isinstance(result, str) else result
            record['evidence_case_corrections'] = [
                {'original_quote': a, 'source_quote': b}
                for a, b in zip(original.get('evidence', []), value.get('evidence', [])) if a != b]
        w.save(path, record)
        return value
