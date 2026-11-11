import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.style_schema import FIELDS, validate_profile, OBSERVE_PROMPT, profile_prompt
from core.style_analyzer import StyleAnalyzer
import workflow_benchmark as w
from nine_dimension_pilot import render, CONDITIONS

class NineTests(unittest.TestCase):
    def test_consumer_preserves_all_nine(self):
        profile = {f: 'value-' + f for f in FIELDS}
        prompt = profile_prompt({'style_summary': {'designer_style_summary': profile}})
        for field in FIELDS:
            self.assertIn(profile[field], prompt)

    def test_exact_schema(self):
        prompts = w.load(w.HERE / 'workflow_prompts.json')
        self.assertEqual(prompts['fields'], FIELDS)
        self.assertEqual(len(FIELDS), 9)
        for field in FIELDS:
            self.assertIn(field, OBSERVE_PROMPT)
        with self.assertRaises(ValueError):
            validate_profile(dict.fromkeys(FIELDS[:-1]))

    def test_no_silent_legacy_fallback(self):
        analyzer = object.__new__(StyleAnalyzer)
        analyzer.llm_client = SimpleNamespace(generate_text_llm=lambda _: '{"designer_style_summary": {}}')
        self.assertIsNone(analyzer.synthesize_style(['{}']))

    def test_six_conditions(self):
        prompts = w.load(w.HERE / 'workflow_prompts.json')
        profile = {field: 'reference-' + field for field in FIELDS}
        task = {'brief': 'blue coat', 'hard_constraints': ['blue'], 'preserve': [],
                'explicit_fields': {FIELDS[6]: 'blue'}, 'reference_images': [{'image_id': 'r'}]}
        texts = {c: render(task, c, profile, 'summary', 'reconciled', prompts) for c in CONDITIONS}
        self.assertEqual(len(set(texts.values())), 6)
        self.assertIn('reconciled', texts['B_reconciled'])
        self.assertIn('reference-' + FIELDS[6], texts['C_no_priority'])
        self.assertNotIn('reference-' + FIELDS[6], texts['C'])
        self.assertNotIn(FIELDS[6], texts['D'])
        self.assertEqual(profile[FIELDS[6]], 'reference-' + FIELDS[6])

if __name__ == '__main__':
    unittest.main()
