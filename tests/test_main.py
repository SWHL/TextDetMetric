# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import sys
from pathlib import Path

import cv2
import yaml
from modelscope.msdatasets import MsDataset
from tqdm import tqdm

cur_dir = Path(__file__).resolve().parent
root_dir = cur_dir.parent

sys.path.append(str(root_dir))

from det_demos.ch_ppocr_v3_det import TextDetector
from text_det_metric import DetectionIoUEvaluator


def read_yaml(yaml_path):
    with open(yaml_path, "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


config_path = root_dir / "det_demos" / "ch_ppocr_v3_det" / "config.yaml"
config = read_yaml(str(config_path))

# Configure the onnx model path.
config["model_path"] = str(root_dir / "det_demos" / config["model_path"])

text_detector = TextDetector(config)


def get_pred(save_pred_path: str):
    test_data = MsDataset.load(
        "text_det_test_dataset",
        namespace="liekkas",
        subset_name="default",
        split="test",
    )

    content = []
    for one_data in tqdm(test_data):
        img_path = one_data.get("image:FILE")

        img = cv2.imread(str(img_path))
        dt_boxes, elapse = text_detector(img)
        content.append(f"{img_path}\t{dt_boxes.tolist()}\t{elapse}")

    with open(save_pred_path, "w", encoding="utf-8") as f:
        for v in content:
            f.write(f"{v}\n")


metric = DetectionIoUEvaluator()


def test_normal():
    pred_path = "1.txt"
    get_pred(str(pred_path))
    result = metric(pred_path)

    assert str(result["precision"]) == "0.6958333333333333"
    assert str(result["recall"]) == "0.8608247422680413"

    Path(pred_path).unlink()
