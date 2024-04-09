# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from text_det_metric import DetectionIoUEvaluator

metric = DetectionIoUEvaluator()

pred_path = "pred.txt"
metric = metric(pred_path)
print(metric)
