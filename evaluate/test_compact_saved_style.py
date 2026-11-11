"""Offline tests; no credentials or network requests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_compact_saved_style as c


class Tests(unittest.TestCase):
    def test_unique_case_alignment_is_exact_and_non_mutating(self):
        raw = {'primary_source': 'image_1', 'description': 'unchanged design', 'evidence': ['image_4: Blue coat']}
        archive = 'Image_4: Blue coat'
        fixed = c.validate_plan(raw, archive, False, normalize_case=True)
        self.assertEqual(fixed['evidence'], [archive])
        self.assertEqual(fixed['description'], raw['description'])
        self.assertEqual(raw['evidence'], ['image_4: Blue coat'])

    def test_case_alignment_does_not_accept_word_punctuation_or_space_changes(self):
        for quote in ('Blue jacket', 'Blue  coat', 'Blue, coat', 'Écoat'):
            value = {'primary_source': 'image_1', 'description': 'coat', 'evidence': [quote]}
            with self.assertRaisesRegex(ValueError, 'quotation'):
                c.validate_plan(value, 'Blue coat and écoat', False, normalize_case=True)

    def test_case_alignment_requires_one_location(self):
        value = {'primary_source': 'image_1', 'description': 'coat', 'evidence': ['BLUE coat']}
        with self.assertRaisesRegex(ValueError, 'quotation'):
            c.validate_plan(value, {'a': 'Blue coat', 'b': 'Blue coat'}, False, normalize_case=True)

    def test_case_alignment_records_original_without_an_extra_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            raw = {'primary_source': 'image_1', 'description': 'coat', 'evidence': ['blue']}
            seen = []
            validate = lambda x: c.validate_plan(c.w.strict_json(x) if isinstance(x, str) else x,
                                                 'Blue coat', False, normalize_case=True)
            with patch.object(c.w, 'Client', self.mock_planner([json.dumps(raw)], seen)):
                c.plan_cache(out/'plan.json', budget, {'vlm_model': 'test'}, 'same', validate)
            saved = c.w.load(out/'plan.json')
            self.assertEqual(len(seen), 1)
            self.assertEqual(saved['evidence_case_corrections'], [{'original_quote': 'blue', 'source_quote': 'Blue'}])

    def test_authorized_truncation_has_single_larger_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            cfg, prompt = {'vlm_model': 'test', 'text_max_tokens': 8192}, 'same'
            signature = c.w.text_hash(json.dumps(['text', cfg, prompt, [], None], sort_keys=True, ensure_ascii=False))
            budget.data['requests'].append({'request_sha256': signature, 'state': 'dispatched', 'accounted_usd': .05})
            c.w.save(out/'calls/request_0001/call_00001.metadata.json',
                     {'http_status': 200, 'requested_model': 'test', 'usage': {'completion_tokens': 8192}})
            good = json.dumps({'primary_source': 'image_1', 'description': 'coat', 'evidence': ['Blue']})
            seen, captured = [], []
            Parent = self.mock_planner([good], seen)
            class Client(Parent):
                def __init__(self, request_cfg, folder):
                    captured.append(request_cfg)
                    super().__init__(request_cfg, folder)
            with patch.object(c.w, 'Client', Client):
                for _ in range(2):
                    c.plan_cache(out/'plan.json', budget, cfg, prompt, self.validate_sample, signature)
            self.assertEqual(len(seen), 1)
            self.assertEqual(captured[0]['text_max_tokens'], 16384)
            self.assertEqual(captured[0]['timeout_seconds'], 360)
            self.assertEqual(c.w.load(out/'plan.json')['selected_attempt'], 2)

    def test_network_failure_cannot_be_called_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            cfg, prompt = {'vlm_model': 'test', 'text_max_tokens': 8192}, 'same'
            signature = c.w.text_hash(json.dumps(['text', cfg, prompt, [], None], sort_keys=True, ensure_ascii=False))
            budget.data['requests'].append({'request_sha256': signature, 'state': 'dispatched', 'accounted_usd': .35})
            with patch.object(c.w, 'Client') as client:
                with self.assertRaisesRegex(ValueError, 'network failures'):
                    c.plan_cache(out/'plan.json', budget, cfg, prompt, self.validate_sample, signature)
                client.assert_not_called()

    def mock_planner(self, responses, seen):
        class Client:
            def __init__(self, cfg, out):
                self.cfg, self.out = cfg, out
            def text(self, prompt, paths):
                seen.append((prompt, paths))
                value = responses.pop(0)
                if isinstance(value, Exception):
                    raise value
                c.w.save(self.out / 'call_00001.text.json',
                         {'prompt': prompt, 'images': [], 'response': value})
                c.w.save(self.out / 'call_00001.metadata.json',
                         {'requested_model': self.cfg['vlm_model'], 'usage': {'cost': .001}})
                return value
        return Client

    def planner_budget(self, out):
        c.w.save(out / 'budget_ledger.json', {'cap_usd': 25, 'prior_probe_usd': 0, 'requests': []})
        return c.Budget(out)

    def validate_sample(self, x):
        return c.validate_plan(c.w.strict_json(x) if isinstance(x, str) else x, 'Blue coat', False)

    def test_planner_retry_unchanged_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            bad = json.dumps({'primary_source': 'image_1', 'description': 'coat', 'evidence': ['blue']})
            good = json.dumps({'primary_source': 'image_1', 'description': 'coat', 'evidence': ['Blue']})
            seen = []
            with patch.object(c.w, 'Client', self.mock_planner([bad, good], seen)):
                for _ in range(2):
                    c.plan_cache(out/'plan.json', budget, {'vlm_model': 'test'}, 'same prompt', self.validate_sample)
            self.assertEqual(seen, [('same prompt', []), ('same prompt', [])])
            self.assertEqual(c.w.load(out/'plan.json')['selected_attempt'], 2)

    def test_planner_retry_limit_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            bad = json.dumps({'primary_source': 'image_1', 'description': 'coat', 'evidence': ['wrong']})
            seen = []
            with patch.object(c.w, 'Client', self.mock_planner([bad, bad], seen)):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, 'retry exhausted'):
                        c.plan_cache(out/'plan.json', budget, {'vlm_model': 'test'}, 'same', self.validate_sample)
            self.assertEqual(len(seen), 2)

    def test_network_error_does_not_enable_planner_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            seen = []
            with patch.object(c.w, 'Client', self.mock_planner([RuntimeError('network')], seen)):
                with self.assertRaisesRegex(RuntimeError, 'network'):
                    c.plan_cache(out/'plan.json', budget, {'vlm_model': 'test'}, 'same', self.validate_sample)
            self.assertEqual(len(seen), 1)

    def test_prior_invalid_original_is_not_called_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.planner_budget(out)
            cfg, prompt = {'vlm_model': 'test'}, 'same'
            signature = c.w.text_hash(json.dumps(['text', cfg, prompt, [], None], sort_keys=True, ensure_ascii=False))
            bad = json.dumps({'primary_source': 'image_1', 'description': 'coat', 'evidence': ['blue']})
            good = json.dumps({'primary_source': 'image_1', 'description': 'coat', 'evidence': ['Blue']})
            seen = []
            with patch.object(c.w, 'Client', self.mock_planner([bad, good], seen)):
                budget.call('text', cfg, prompt, [], None, signature)
                c.plan_cache(out/'plan.json', budget, cfg, prompt, self.validate_sample)
            self.assertEqual(len(seen), 2)

    def test_evaluation_roles_match_four_references(self):
        text = c.evaluation_roles('图1和图2为服装风格参考，图3为生成结果。其余评价规则不变')
        self.assertTrue(text.startswith('图1至图4为服装风格参考，图5为生成结果。'))
        self.assertNotIn('图3为生成结果', text)

    def test_budget_ceiling_blocks_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c.w.save(out / 'budget_ledger.json', {'cap_usd': 25, 'prior_probe_usd': 24.95, 'requests': []})
            budget = c.Budget(out)
            with patch.object(c.w, 'Client') as client:
                with self.assertRaisesRegex(ValueError, 'ceiling'):
                    budget.call('image', {}, 'test', [], 1, 'unique')
                client.assert_not_called()

    def test_failed_dispatch_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c.w.save(out / 'budget_ledger.json', {'cap_usd': 25, 'prior_probe_usd': 0,
                'requests': [{'request_sha256': 'seen', 'accounted_usd': .35}]})
            with self.assertRaisesRegex(ValueError, 'no retry'):
                c.Budget(out).call('text', {}, 'test', [], None, 'seen')

    def test_profile_phase_respects_three_dollars(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            c.w.save(out / 'budget_ledger.json', {'cap_usd': 25, 'prior_probe_usd': 2.9, 'requests': []})
            with self.assertRaisesRegex(ValueError, 'ceiling'):
                c.Budget(out, 3).call('text', {}, 'test', [], None, 'new')

    def test_cache_avoids_second_call_and_checks_signature(self):
        class Fake:
            count = 0
            def call(self, *args):
                self.count += 1
                return {'ok': 1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.json'
            fake = Fake()
            for _ in range(2):
                self.assertEqual(c.cache(path, fake, 'text', {}, 'prompt', [], lambda x: x), {'ok': 1})
            self.assertEqual(fake.count, 1)
            with self.assertRaisesRegex(ValueError, 'changed'):
                c.cache(path, fake, 'text', {}, 'different', [], lambda x: x)

    def test_planner_quotes_must_be_verbatim(self):
        plan = {'primary_source': 'image_1', 'description': 'blue coat', 'evidence': ['blue']}
        self.assertEqual(c.validate_plan(plan, 'blue coat', False), plan)
        plan['evidence'] = ['invented']
        with self.assertRaisesRegex(ValueError, 'quotation'):
            c.validate_plan(plan, 'blue coat', False)

    def test_nine_schema_cannot_be_replaced_by_prose(self):
        with self.assertRaises(ValueError):
            c.validate_plan({'primary_source': 'image_1', 'description': 'text', 'evidence': ['blue']}, 'blue', True)

    def test_task_spec_has_zero_generation_images(self):
        out = Path(__file__).resolve().parents[1] / 'reproduction/compact_study'
        spec = c.w.load(out / 'tasks.candidate.json')
        self.assertEqual(len(spec['tasks']), 30)
        for task in spec['tasks']:
            self.assertFalse(task['generation_image_inputs'])
            self.assertTrue(set(task['eligible_style_dimensions']).isdisjoint(task['style_excluded_dimensions']))


if __name__ == '__main__':
    unittest.main()
