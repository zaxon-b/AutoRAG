from __future__ import annotations

import unittest
from unittest.mock import patch

from models.manual import ManualChunk
from service.icon_grouping_service import _review_partition
from service.visual_object_naming_service import (
    _candidate_names,
    _validate_icon_name,
    _validate_operation_name,
)
from service.visual_region_extractor import _join_wrapped_name_candidate


class VisualObjectNamingTest(unittest.TestCase):
    def test_icon_name_must_be_a_literal_manual_candidate(self):
        candidates = ["动力电池电量低警告灯", "转向灯指示灯"]

        self.assertEqual(_validate_icon_name("转向灯指示灯", candidates), "转向灯指示灯")
        self.assertIsNone(_validate_icon_name("左转向灯", candidates))
        self.assertIsNone(_validate_icon_name(None, candidates))

    def test_icon_vlm_only_receives_the_direct_spatial_label(self):
        payload = {
            "modality": "icon",
            "_name_candidates": ["ABS 故障警告灯", "智能钥匙系统警告灯", "发动机故障警告灯"],
        }

        self.assertEqual(_candidate_names(payload), ["ABS 故障警告灯"])

    def test_wrapped_manual_icon_name_is_joined_before_vlm(self):
        self.assertEqual(
            _join_wrapped_name_candidate(["自动紧急制动故障警告", "灯*", "邻行名称"]),
            ["自动紧急制动故障警告灯*", "邻行名称"],
        )
        self.assertEqual(
            _join_wrapped_name_candidate(["动力系统故障警告灯", "冷却液温度过高警告灯"]),
            ["动力系统故障警告灯", "冷却液温度过高警告灯"],
        )

    def test_operation_name_rejects_generic_callout_names(self):
        self.assertEqual(
            _validate_operation_name("EV纯电动工作模式", "EV模式下由动力电池纯电驱动车辆"),
            "EV纯电动工作模式",
        )
        self.assertIsNone(_validate_operation_name("固定锚支座", "EV纯电动工作模式"))
        self.assertIsNone(_validate_operation_name("温馨提示"))
        self.assertIsNone(_validate_operation_name("示意图"))
        self.assertIsNone(_validate_operation_name("null"))


class IconGroupReviewTest(unittest.TestCase):
    def setUp(self):
        self.rows = {
            "a": ManualChunk(chunk_id="a", manual_id="m", page_number=1, modality="icon"),
            "b": ManualChunk(chunk_id="b", manual_id="m", page_number=2, modality="icon"),
        }

    @patch("service.icon_grouping_service._vlm_partition")
    @patch("service.icon_grouping_service._clean_env", return_value="test-key")
    def test_high_confidence_vlm_partition_merges_units(self, _key, partition):
        partition.return_value = [{"indices": [1, 2], "confidence": 0.95}]

        self.assertEqual(_review_partition(["a", "b"], self.rows), [["a", "b"]])

    @patch("service.icon_grouping_service._vlm_partition")
    @patch("service.icon_grouping_service._clean_env", return_value="test-key")
    def test_low_confidence_vlm_partition_keeps_units_separate(self, _key, partition):
        partition.return_value = [{"indices": [1, 2], "confidence": 0.2}]

        self.assertEqual(_review_partition(["a", "b"], self.rows), [["a"], ["b"]])


if __name__ == "__main__":
    unittest.main()
