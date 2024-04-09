# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent

engine = RapidOCR()

dataset = load_dataset("SWHL/text_det_test_dataset")
test_data = dataset["test"]

content = []
for i, one_data in enumerate(tqdm(test_data)):
    img = np.array(one_data.get("image"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    dt_boxes, elapse = engine(img, use_det=True, use_cls=False, use_rec=False)

    dt_boxes = [] if dt_boxes is None else dt_boxes
    elapse = 0 if elapse is None else elapse[0]

    gt_boxes = [v["points"] for v in one_data["shapes"]]
    content.append(f"{dt_boxes}\t{gt_boxes}\t{elapse}")

with open("pred.txt", "w", encoding="utf-8") as f:
    for v in content:
        f.write(f"{v}\n")
