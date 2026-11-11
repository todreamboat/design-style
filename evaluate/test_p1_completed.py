import unittest
import analyze_p1_completed as a


class AuditTests(unittest.TestCase):
    def test_constant_labels_kappa_not_invented(self):
        result=a.agreement([{'human':'met','machine':'met'}]*4,['met','violated','unknown'])
        self.assertEqual(result['exact_agreement'],1.)
        self.assertIsNone(result['kappa'])

    def test_constant_human_zero_kappa(self):
        result=a.agreement([{'human':'met','machine':'met'}]*3+[{'human':'met','machine':'violated'}],['met','violated'])
        self.assertEqual(result['kappa'],0.)

    def test_weighted_ordinal_order(self):
        result=a.agreement([{'human':'partial','machine':'partial'},
            {'human':'retained','machine':'retained'},{'human':'not_retained','machine':'not_retained'}],
            ['not_retained','partial','retained'],True)
        self.assertEqual(result['kappa'],1.)

    def test_duplicate_input_rejected(self):
        with self.assertRaises(ValueError):
            a.no_duplicates([('case_001',{}),('case_001',{})])

    @unittest.skipUnless((a.p.OUT/'human_audit_analysis.json').exists(),
                         'Requires original human-audit artifacts, not distributed in the public snapshot')
    def test_actual_coverage_and_disagreements(self):
        data=a.w.load(a.p.OUT/'human_audit_analysis.json')
        self.assertEqual(data['cases'],48)
        self.assertEqual(len(data['disagreements']),50)
        self.assertEqual(data['agreement']['style_all_labels']['judgments'],216)
        self.assertEqual(data['agreement']['task']['judgments'],96)
        flat=a.w.load(a.p.OUT/'flat_ablation_analysis.json')
        self.assertEqual(flat['outputs'],24)
        self.assertAlmostEqual(flat['summary']['style'],.925)
        self.assertLessEqual(flat['accounted_or_reserved_usd'],5.)


if __name__=='__main__':
    unittest.main()
