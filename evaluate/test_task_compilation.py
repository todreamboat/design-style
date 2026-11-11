import copy
import unittest
import os
import run_task_compilation as c


class CompilationTests(unittest.TestCase):
    def test_process_guard_recognizes_current_process(self):
        self.assertTrue(c.process_running(os.getpid()))

    def test_process_guard_accepts_nonexistent_pid(self):
        self.assertFalse(c.process_running(2147483647))

    def setUp(self):
        self.profile=dict.fromkeys(c.FIELDS)
        self.profile[c.FIELDS[2]]='image_1: white shirt under a black coat'
        self.task=dict(task_id='main_b1_structure',hard_constraints=['belt'],
                       explicit_fields={c.FIELDS[4]:'visible belt'})
        self.plan=dict(primary_source='image_1',designer_style_summary=copy.deepcopy(self.profile),
            evidence={f:([v] if v else []) for f,v in self.profile.items()},task_modified_fields=[])

    def test_valid_plan(self):
        self.assertIs(c.validate_plan(self.plan,self.profile,self.task),self.plan)

    def test_fabricated_quote_rejected(self):
        self.plan['evidence'][c.FIELDS[2]]=['silk coat']
        with self.assertRaises(ValueError): c.validate_plan(self.plan,self.profile,self.task)

    def test_unknown_source_rejected(self):
        self.plan['primary_source']='image_99'
        with self.assertRaises(ValueError): c.validate_plan(self.plan,self.profile,self.task)

    def test_unknown_field_invention_rejected(self):
        self.plan['designer_style_summary'][c.FIELDS[5]]='necklace'
        with self.assertRaises(ValueError): c.validate_plan(self.plan,self.profile,self.task)

    def test_explicit_task_change_allowed(self):
        self.plan['designer_style_summary'][c.FIELDS[4]]='visible belt'
        self.plan['task_modified_fields']=[c.FIELDS[4]]
        c.validate_plan(self.plan,self.profile,self.task)

    def test_selected_source_unobservable_field_may_be_null(self):
        field=c.FIELDS[5]
        self.profile[field]='image_1未观察到配饰信息；image_2有金属扣'
        self.plan['evidence'][field]=['image_1未观察到配饰信息']
        self.plan['designer_style_summary'][field]=None
        c.validate_plan(self.plan,self.profile,self.task)

    def test_known_field_cannot_be_silently_dropped(self):
        self.plan['designer_style_summary'][c.FIELDS[2]]=None
        with self.assertRaises(ValueError): c.validate_plan(self.plan,self.profile,self.task)

    def test_wrapper_same_for_both_conditions(self):
        text=c.generation_prompt(self.task,self.profile)
        self.assertIn(self.profile[c.FIELDS[2]],text)
        self.assertNotIn('primary_source',text)
        self.assertNotIn('evidence',text)


if __name__=='__main__': unittest.main()
