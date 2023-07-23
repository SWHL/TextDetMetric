# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from .utils import (
    DBPostProcess,
    check_and_read_gif,
    create_operators,
    get_image_file_list,
    transform,
)

cur_dir = Path(__file__).resolve().parent
DEFAULT_MODEL = cur_dir / "ch_PP-OCRv3_det_infer.onnx"


class TextDetector:
    def __init__(self, det_model_path: str = str(DEFAULT_MODEL)):
        pre_process_list = [
            {"DetResizeForTest": {"limit_side_len": 736, "limit_type": "min"}},
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = DBPostProcess(
            thresh=0.3,
            box_thresh=0.5,
            max_candidates=1000,
            unclip_ratio=1.8,
            use_dilation=True,
        )

        sess_opt = onnxruntime.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        self.session = onnxruntime.InferenceSession(det_model_path, sess_opt)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0, 0

        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)
        shape_list = np.expand_dims(shape_list, axis=0)

        starttime = time.time()
        preds = self.session.run([self.output_name], {self.input_name: img})

        post_result = self.postprocess_op(preds[0], shape_list)
        dt_boxes, scores = post_result[0]["points"], post_result[0]["scores"]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, scores, elapse


def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=str, default=None, help="image_path|image_dir"
    )
    parser.add_argument("--model_path", type=str, default=None, help="model_path")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    text_detector = TextDetector(args.model_path)

    img_file_list = get_image_file_list(args.image_dir)

    result = []
    for image_path in img_file_list:
        img, flag = check_and_read_gif(image_path)
        if not flag:
            img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"error in loading image:{image_path}")

        dt_boxes, elapse = text_detector(img)

        src_im = draw_text_det_res(dt_boxes, image_path)
        cv2.imwrite(f"vis/{image_path}", src_im)

        new_dt_boxes = list(map(lambda x: x.astype(np.int).tolist(), dt_boxes))
        result.append(f"{Path(image_path).name}\t{str(new_dt_boxes)}")

    if args.output_path:
        output_path = Path(args.image_dir).parent / args.output_path
        with open(str(output_path), "w", encoding="utf-8") as f:
            for v in result:
                f.write(v + "\n")
        print(f"{output_path}已经保存！")

    # region: 可视化单张图像效果
    # src_im = draw_text_det_res(dt_boxes, args.image_path)
    # cv2.imwrite('det_results.jpg', src_im)
    # print('图像已经保存为det_results.jpg了')
    # endregion
