# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from pathlib import Path

import cv2
import yaml
from modelscope.msdatasets import MsDataset
from tqdm import tqdm

from det_demos.ch_ppocr_v3_det import TextDetector

root_dir = Path(__file__).resolve().parent


def read_yaml(yaml_path):
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


test_data = MsDataset.load(
    "text_det_test_dataset",
    namespace="liekkas",
    subset_name="default",
    split="test",
)

config_path = root_dir / "det_demos" / "ch_ppocr_v3_det" / "config.yaml"
config = read_yaml(str(config_path))

# Configure the onnx model path.
config["model_path"] = str(root_dir / "det_demos" / config["model_path"])

text_detector = TextDetector(config)

content = []
for one_data in tqdm(test_data):
    img_path = one_data.get("image:FILE")

    img = cv2.imread(str(img_path))
    dt_boxes, elapse = text_detector(img)
    content.append(f"{img_path}\t{dt_boxes.tolist()}\t{elapse}")

with open("pred.txt", "w", encoding="utf-8") as f:
    for v in content:
        f.write(f"{v}\n")
