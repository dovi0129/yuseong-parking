"""Known-answer rule, EXIF, original-pixel, and label-isolation tests."""
from datetime import datetime,timedelta
import io
import unittest
from unittest.mock import patch

import piexif
from PIL import Image

from .generate import exif_bytes,offset
from .review import box_to_raw,decide_pair,distance_m,normalize_plate,read_exif,review_photo_pair


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.gps=(36.3567,127.3486)
        self.first={'plate':'123가4567','box':[10,20,100,45],'taken':'2024-07-01 12:00:00',
                    'datetime':datetime(2024,7,1,12),'gps':self.gps}
        self.second={**self.first,'taken':'2024-07-01 12:01:00','datetime':self.first['datetime']+timedelta(seconds=60),
                     'gps':offset(self.gps,15)}

    def decide(self,second=None,**kwargs):
        return decide_pair(self.first,second or self.second,*self.gps,**kwargs)

    def test_inclusive_60_seconds_and_15_metres(self):
        self.assertEqual(self.decide()['verdict'],'요건 충족')
        self.assertAlmostEqual(self.decide()['move_m'],15,places=4)

    def test_gap_59_seconds_rejected(self):
        result=self.decide({**self.second,'datetime':self.first['datetime']+timedelta(seconds=59)})
        self.assertEqual(result['reasons'],['촬영 간격 미달'])

    def test_upload_order_does_not_change_capture_gap(self):
        result=decide_pair(self.second,self.first,*self.gps)
        self.assertEqual(result['verdict'],'요건 충족')

    def test_capture_location_reason_is_separate(self):
        result=self.decide({**self.second,'gps':offset(self.gps,20)})
        self.assertEqual(result['reasons'],['촬영 위치 불일치'])

    def test_report_location_reason_is_separate(self):
        a={**self.first,'gps':offset(self.gps,80)}
        b={**self.second,'gps':offset(self.gps,85)}
        result=decide_pair(a,b,*self.gps)
        self.assertEqual(result['reasons'],['신고 위치 불일치'])

    def test_farthest_photo_sets_report_distance(self):
        b={**self.second,'gps':offset(self.gps,51)}
        result=self.decide(b,max_move_m=60)
        self.assertAlmostEqual(result['report_m'],51,places=4)
        self.assertEqual(result['reasons'],['신고 위치 불일치'])

    def test_report_distance_50_inclusive(self):
        b={**self.second,'gps':offset(self.gps,50)}
        self.assertEqual(self.decide(b,max_move_m=60)['verdict'],'요건 충족')

    def test_unreadable_cannot_satisfy_same_vehicle(self):
        self.assertIn('번호 판독 불가',self.decide({**self.second,'plate':None})['reasons'])

    def test_mismatched_plate(self):
        self.assertEqual(self.decide({**self.second,'plate':'12나3456'})['reasons'],['번호 불일치'])

    def test_missing_gps_is_not_zero_distance(self):
        result=self.decide({**self.second,'gps':None})
        self.assertEqual(result['reasons'],['위치 정보 없음'])
        self.assertIsNone(result['move_m']);self.assertIsNone(result['report_m'])

    def test_missing_report_cannot_be_fully_satisfied(self):
        self.assertIn('신고 위치 정보 없음',decide_pair(self.first,self.second)['reasons'])

    def test_missing_capture_time(self):
        self.assertIn('촬영 시각 정보 없음',self.decide({**self.second,'datetime':None})['reasons'])

    def test_invalid_threshold_rejected(self):
        for value in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):self.decide(min_gap_s=value)

    def test_private_plate_format(self):
        self.assertEqual(normalize_plate('１２３ 가 ４５６７'),'123가4567')
        for text in ['123하4567','123가456','abcdef','123가45678']:
            self.assertIsNone(normalize_plate(text))

    def test_exif_roundtrip(self):
        image=Image.new('RGB',(100,100));image.info['exif']=exif_bytes(self.first['datetime'],self.gps)
        metadata=read_exif(image)
        self.assertEqual(metadata['datetime'],self.first['datetime'])
        self.assertLess(distance_m(metadata['gps'],self.gps),.001)

    def test_invalid_time_does_not_discard_gps(self):
        exif=piexif.load(exif_bytes(self.first['datetime'],self.gps))
        exif['Exif'][piexif.ExifIFD.DateTimeOriginal]=b'not-a-capture-time'
        image=Image.new('RGB',(100,100));image.info['exif']=piexif.dump(exif)
        metadata=read_exif(image)
        self.assertIsNone(metadata['datetime']);self.assertIsNotNone(metadata['gps'])

    def test_south_west_exif_signs(self):
        image=Image.new('RGB',(100,100));image.info['exif']=exif_bytes(self.first['datetime'],(-36.,-127.))
        self.assertEqual(read_exif(image)['gps'],(-36.,-127.))

    def test_missing_gps_reference_is_unavailable_not_assumed_north(self):
        exif=piexif.load(exif_bytes(self.first['datetime'],self.gps));del exif['GPS'][piexif.GPSIFD.GPSLatitudeRef]
        image=Image.new('RGB',(100,100));image.info['exif']=piexif.dump(exif)
        self.assertIsNone(read_exif(image)['gps'])

    def test_original_pixel_box_for_orientation_6(self):
        self.assertEqual(box_to_raw([10,20,40,50],6,100,60),[20,20,50,50])

    def test_filename_cannot_supply_recognition(self):
        buffers=[]
        for dt in [self.first['datetime'],self.second['datetime']]:
            buffer=io.BytesIO();buffer.name='999나9999.jpg'
            Image.new('RGB',(200,100),'white').save(buffer,format='JPEG',exif=exif_bytes(dt,self.gps));buffer.seek(0);buffers.append(buffer)
        with patch('src.photo.review.read_plate',return_value={'plate':None,'box':None,'confidence':None}):
            result=review_photo_pair(*buffers,*self.gps)
        self.assertIsNone(result['plate_1']);self.assertEqual(result['verdict'],'보완 필요')


if __name__=='__main__':unittest.main()
