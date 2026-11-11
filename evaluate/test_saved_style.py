import unittest
from unittest.mock import Mock
import run_saved_style as s


class SavedStyleTests(unittest.TestCase):
    def setUp(self):
        self.task = {'task_id':'main_b1_appearance','hard_constraints':['purple'],
                     'reference_images':[{'path':'must-not-send.jpg'}]}
        self.saved = {'profile':{'cut':'asymmetric'},'summary':'plain prose'}

    def test_task_only_contains_no_saved_style(self):
        prompt = s.render(self.task,'task_only',self.saved)
        self.assertNotIn('asymmetric',prompt)
        self.assertNotIn('plain prose',prompt)
        self.assertNotIn('must-not-send',prompt)

    def test_distinct_saved_inputs(self):
        self.assertIn('plain prose',s.render(self.task,'saved_summary',self.saved))
        self.assertIn('asymmetric',s.render(self.task,'saved_nine',self.saved))

    def test_generation_has_zero_images(self):
        budget = Mock()
        s.image_call(budget,{},dict(input_records=[],prompt='text',seed=1))
        budget.call.assert_called_once_with('image',{},'text',[],1)

    def test_generation_rejects_image_leakage(self):
        budget = Mock()
        with self.assertRaises(ValueError):
            s.image_call(budget,{},dict(input_records=['image'],prompt='text',seed=1))
        budget.call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
