# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
from modelscope.msdatasets import MsDataset

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from ch_mobile_v2_det import TextDetector
from text_det_metric import DetectionIoUEvaluator


def get_pred(save_pred_path: str):
    test_data = MsDataset.load(
        "text_det_test_dataset",
        namespace="liekkas",
        subset_name="default",
        split="test",
    )

    text_detector = TextDetector()

    content = []
    for one_data in test_data:
        img_path = one_data.get("image:FILE")
        img = cv2.imread(str(img_path))
        dt_boxes, scores, _ = text_detector(img)
        content.append(f"{img_path}\t{dt_boxes.tolist()}\t{scores}")

    with open(save_pred_path, "w", encoding="utf-8") as f:
        for v in content:
            f.write(f"{v}\n")


metric = DetectionIoUEvaluator()


def test_normal():
    pred_path = "1.txt"
    get_pred(str(pred_path))
    result = metric(pred_path)

    assert str(result["precision"]) == "0.6926406926406926"
    assert str(result["recall"]) == "0.8247422680412371"

    Path(pred_path).unlink()
