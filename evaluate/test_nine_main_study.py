import unittest
from pathlib import Path
import tempfile
import workflow_benchmark as w
from nine_main_study import CONDITIONS, StudyBudget, literature_prompt, render, EXCLUDED_BRANDS

class MainStudyTests(unittest.TestCase):
    def test_eight_distinct_arms(self):
        p=w.load(w.HERE/'workflow_prompts.json')
        profile={f:'ref '+f for f in p['fields']}
        task={'brief':'purple coat','hard_constraints':['purple'],'preserve':[],
              'explicit_fields':{p['fields'][6]:'purple'},'reference_images':[{'image_id':'r1'}]}
        literature={'L_no_persona':'no persona','L_multi_expert':'two-stage expert'}
        prompts=[render(task,c,profile,'summary','reconciled',p,literature) for c in CONDITIONS]
        self.assertEqual(len(set(prompts)),8)
        self.assertTrue(all('r1' in text and 'purple coat' in text for text in prompts))

    def test_original_role_order_is_retained(self):
        task={'brief':'coat','hard_constraints':['purple']}
        self.assertIn('服装设计师',literature_prompt('designer',task,'reference'))
        self.assertIn('AI图像生成专家',literature_prompt('expert',task,'reference','stage1'))
        self.assertIn('stage1',literature_prompt('expert',task,'reference','stage1'))
        self.assertNotIn('你是一位',literature_prompt('none',task,'reference'))

    def test_budget_and_development_exclusion(self):
        self.assertTrue({'16Arlington','224 Hommes'}.issubset(EXCLUDED_BRANDS))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                StudyBudget(Path(d),21)
            budget=StudyBudget(Path(d),.01)
            with self.assertRaises(ValueError):
                budget.call('image',{},'not sent',[])

if __name__=='__main__':
    unittest.main()
