"""Offline retry policy checks; no credentials, network, or real sleeps."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import compact_retry as r
import run_compact_saved_style as c


CFG = {'vlm_model': 'test', 'image_model': 'test-image',
       'text_max_tokens': 8192, 'timeout_seconds': 180}


class Tests(unittest.TestCase):
    def budget(self, out, spent=0):
        c.w.save(out/'budget_ledger.json', {'cap_usd': 25, 'prior_probe_usd': spent, 'requests': []})
        return r.AutoBudget(c.Budget(out), sleep=lambda n: None)

    def client(self, actions, calls):
        class Client:
            def __init__(self, cfg, out):
                self.cfg, self.out = cfg, out
            def text(self, prompt, paths):
                calls.append((prompt, paths, self.cfg))
                action = actions.pop(0)
                if action == 'network':
                    c.w.save(self.out/'call_00001.error.json',
                             {'error': 'network_error', 'exception_chain_types': ['ChunkedEncodingError']})
                    raise RuntimeError('Network error')
                if action == '401':
                    c.w.save(self.out/'call_00001.error.json', {'http_status': 401})
                    raise RuntimeError('HTTP 401')
                if action == 'length':
                    c.w.save(self.out/'call_00001.metadata.json', {'http_status': 200,
                        'usage': {'completion_tokens': self.cfg['text_max_tokens'], 'cost': .05}})
                    c.w.save(self.out/'call_00001.text.json', {'prompt': prompt, 'response': '{',
                        'images': [], 'finish_reason': 'length'})
                    raise ValueError('Text truncated by token limit')
                c.w.save(self.out/'call_00001.metadata.json', {'http_status': 200, 'usage': {'cost': .001}})
                c.w.save(self.out/'call_00001.text.json', {'prompt': prompt, 'images': [], 'response': action})
                return action
        return Client

    def test_network_retries_fixed_payload_and_preserves_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = self.budget(Path(tmp))
            calls = []
            with patch.object(r, 'DurableClient', self.client(['network', 'ok'], calls)):
                self.assertEqual(budget.call('text', CFG, 'same', [], None, 'sig'), 'ok')
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][:2], calls[1][:2])
            self.assertAlmostEqual(budget.spent(), .351)

    def test_prior_network_failure_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.budget(out)
            budget.data['requests'].append({'request_sha256': 'sig', 'state': 'dispatched',
                                            'accounted_usd': .35, 'kind': 'text'})
            c.w.save(out/'calls/request_0001/call_00001.error.json', {'error': 'network_error'})
            calls = []
            with patch.object(r, 'DurableClient', self.client(['ok'], calls)):
                self.assertEqual(budget.call('text', CFG, 'same', [], None, 'sig'), 'ok')
            self.assertEqual(len(calls), 1)

    def test_auth_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = self.budget(Path(tmp))
            calls = []
            with patch.object(r, 'DurableClient', self.client(['401'], calls)):
                with self.assertRaisesRegex(RuntimeError, 'Non-retryable'):
                    budget.call('text', CFG, 'same', [], None, 'sig')
            self.assertEqual(len(calls), 1)

    def test_length_recovery_increases_budget_not_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = self.budget(Path(tmp))
            calls = []
            with patch.object(r, 'DurableClient', self.client(['length', 'ok'], calls)):
                self.assertEqual(budget.call('text', CFG, 'same', [], None, 'sig'), 'ok')
            self.assertEqual(calls[1][2]['text_max_tokens'], 16384)
            self.assertEqual(calls[0][0], calls[1][0])

    def test_invalid_json_retries_but_zero_score_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            budget = self.budget(out)
            calls = []
            validate = lambda x: json.loads(x) if isinstance(x, str) else x
            with patch.object(r, 'DurableClient', self.client(['bad json', '{"score":0}'], calls)):
                value = r.validated_cache(out/'score.json', budget, 'text', CFG, 'same', [], validate)
                again = r.validated_cache(out/'score.json', budget, 'text', CFG, 'same', [], validate)
            self.assertEqual(value, {'score': 0})
            self.assertEqual(again, value)
            self.assertEqual(len(calls), 2)

    def test_ceiling_blocks_additional_network_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = self.budget(Path(tmp), spent=24.9)
            with patch.object(r, 'DurableClient') as client:
                with self.assertRaisesRegex(ValueError, 'budget ceiling'):
                    budget.call('text', CFG, 'same', [], None, 'sig')
                client.assert_not_called()

    def test_generator_rejects_history_images_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = self.budget(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'forbidden'):
                budget.call('image', CFG, 'same', ['not-uploaded.jpg'], 123, 'sig')


if __name__ == '__main__':
    unittest.main()
