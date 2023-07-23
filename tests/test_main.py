# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from text_det_metric import DetectionIoUEvaluator

metric = DetectionIoUEvaluator()
test_file_dir = cur_dir / "test_files"


def test_normal():
    pred_path = test_file_dir / "1.txt"
    result = metric(pred_path)

    assert str(result["precision"]) == "0.6926406926406926"
    assert str(result["recall"]) == "0.8247422680412371"
