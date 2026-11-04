import unittest

import numpy as np
import pandas as pd

from .key_numbers import COLUMNS,concentration_point,repetition_counts,priority_rate,paired_difference,map_review_counts,trimmed_location_count,photo_decision_counts,real_photo_counts


class KeyNumberTests(unittest.TestCase):
    def test_top_ten_percent_uses_ceiling(self):
        counts=np.array([10,5]+[1]*9)
        frame=pd.DataFrame({"spot_fraction":np.arange(1,12)/11,
            "cumulative_fraction":counts.cumsum()/counts.sum(),"n_spots":[11]*11})
        count,share=concentration_point(frame)
        self.assertEqual(count,2)
        self.assertAlmostEqual(share,.625)

    def test_repeated_share_includes_first_record_in_repeated_group(self):
        frame=pd.DataFrame({"reports_per_spot_day":[1,2,3],"n_spot_days":[4,1,2],"n_reports":[4,2,6]})
        total,repeated,share=repetition_counts(frame)
        self.assertEqual((total,repeated),(12,8))
        self.assertAlmostEqual(share,2/3)

    def test_priority_coverage_is_ratio_of_sums(self):
        frame=pd.DataFrame({"captured":[1,9],"total":[1,99]})
        self.assertAlmostEqual(priority_rate(frame),.1)

    def test_map_review_counts_completion_not_unchecked_accuracy(self):
        review=pd.DataFrame({"spot_id":["a","b","c"],"review_status":["map_checked","map_checked","pending"]})
        self.assertEqual(map_review_counts(review),(3,2))

    def test_location_count_only_removes_edge_whitespace(self):
        self.assertEqual(trimmed_location_count(pd.Series([" A","A ","A B","AB",None])),3)

    def test_photo_balanced_accuracy_does_not_follow_class_imbalance(self):
        expected=["요건 충족"]*2+["보완 필요"]*8
        predicted=["요건 충족","보완 필요"]+["보완 필요"]*8
        frame=pd.DataFrame({"expected_verdict":expected,"verdict":predicted,
            "case":["정상"]*2+["간격 미달"]*8,
            "plate_1":["123가4567"]*10,"plate_2":["123가4567"]*10,
            "plate_true_1":["123가4567"]*10,"plate_true_2":["123가4567"]*10})
        result=photo_decision_counts(frame)
        self.assertEqual((result['n_normal'],result['n_reject'],result['normal_accepted'],result['reject_detected']),(2,8,1,8))
        self.assertAlmostEqual(result['verdict_accuracy'],.9)
        self.assertAlmostEqual(result['balanced_accuracy'],.75)

    def test_real_photo_unreadable_is_not_the_same_as_wrong_reading(self):
        frame=pd.DataFrame({"truth_status":["수기 확인"]*3+["번호판 가림"],
            "plate_true":["a","b","c",None],"plate":["a",None,"wrong",None]})
        self.assertEqual(real_photo_counts(frame),(3,1,1))

    def test_paired_delta_keeps_sign_and_requires_matched_dates(self):
        daily=pd.DataFrame({"scenario":["case"]*4,"method":["b3","optimized","optimized","b3"],
            "date":["2024-10-02","2024-10-04","2024-10-02","2024-10-04"],"caught":[1,0,4,5]})
        value,days=paired_difference(daily,"case","b3","caught")
        self.assertEqual((value,days),(-1.,2))
        with self.assertRaises(ValueError):paired_difference(daily.iloc[:-1],"case","b3","caught")
        self.assertEqual(COLUMNS,["항목","값","단위","정의","기간·조건","근거 파일"])


if __name__=="__main__":unittest.main()
