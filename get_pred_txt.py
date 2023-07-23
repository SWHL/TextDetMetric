# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import cv2
from modelscope.msdatasets import MsDataset

from ch_mobile_v2_det import TextDetector

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

with open("pred.txt", "w", encoding="utf-8") as f:
    for v in content:
        f.write(f"{v}\n")
