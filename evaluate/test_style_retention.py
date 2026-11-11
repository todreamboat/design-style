"""Offline regression tests for detail-preserving aggregation and original layers."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.style_schema import FIELDS, synthesis_prompt, validate_synthesis, profile_prompt
from core.style_analyzer import StyleAnalyzer


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.profile = dict.fromkeys(FIELDS)
        self.profile[FIELDS[1]] = 'Houndstooth coat over a white ruffled shirt; zebra-print fitted trousers.'
        self.profile[FIELDS[2]] = 'Fine houndstooth coat; white shirt with ruffled front and cuffs.'
        self.profile[FIELDS[3]] = 'Grey-white zebra-print fitted trousers.'
        self.analyses = [json.dumps(self.profile)]
        self.summary = dict(designer_style_summary=copy.deepcopy(self.profile),
                            overall_style_description='Pattern contrast with layered garments.',
                            style_keywords=['houndstooth', 'ruffled shirt'])

    def test_prompt_preserves_evidence_and_source_ids(self):
        text = synthesis_prompt(self.analyses)
        for value in self.profile.values():
            if value:
                self.assertIn(value, text)
        self.assertIn('image_1', text)
        self.assertIn('not shared by every image', text)
        self.assertIn('not just the overall description', text)

    def test_supported_field_cannot_become_null(self):
        bad = copy.deepcopy(self.summary)
        bad['designer_style_summary'][FIELDS[2]] = None
        with self.assertRaises(ValueError):
            validate_synthesis(bad, self.analyses)

    def test_unknown_in_other_image_does_not_erase_observation(self):
        unknown = json.dumps(dict.fromkeys(FIELDS))
        self.assertIs(validate_synthesis(self.summary, self.analyses+[unknown]), self.summary)

    def test_unobserved_field_cannot_gain_invented_description(self):
        bad = copy.deepcopy(self.summary)
        bad['designer_style_summary'][FIELDS[5]] = 'gold necklace'
        with self.assertRaises(ValueError):
            validate_synthesis(bad,self.analyses)

    def test_bad_summary_shape_is_rejected(self):
        for key in ('overall_style_description','style_keywords'):
            bad = copy.deepcopy(self.summary)
            bad[key] = None
            with self.assertRaises(ValueError):
                validate_synthesis(bad,self.analyses)

    def test_save_load_original_hierarchy_preserves_details(self):
        analyzer = object.__new__(StyleAnalyzer)
        analyzer.vlm_client = SimpleNamespace(model_name='offline')
        analyzer.llm_client = SimpleNamespace(model_name='offline')
        result = dict(designer_info=dict(name='test',hash='id',total_images=1,
            analyzed_images=1,analysis_date='offline'),individual_analyses=self.analyses,
            style_analysis=self.summary,individual_analysis_sources=[{'image_id':'image_1','filename':'coat.jpg'}],
            metadata=dict(analyzer_version='test',analysis_dimensions=9))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'style_analysis.json'
            self.assertTrue(analyzer.save_analysis_result(result,str(path)))
            full = json.loads(path.read_text(encoding='utf-8'))
            saved = json.loads((path.parent/'train.json').read_text(encoding='utf-8'))
            self.assertEqual(full['individual_analyses'],self.analyses)
            self.assertEqual(full['style_analysis'],saved['style_summary'])
            self.assertEqual(saved['metadata']['observation_sources'],result['individual_analysis_sources'])
            prompt = profile_prompt(saved)
            for value in self.profile.values():
                if value:
                    self.assertIn(value,prompt)


if __name__ == '__main__':
    unittest.main()
