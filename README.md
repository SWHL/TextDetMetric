English | [简体中文](https://github.com/SWHL/TextDetMetric/blob/main/docs/README_zh.md)

## Text Detect Metric
<p align="left">
     <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
     <a href=""><img src="https://img.shields.io/badge/python->=3.6,<3.12-aff.svg"></a>
     <a href="https://pypi.org/project/text_det_metric/"><img alt="PyPI" src="https://img.shields.io/pypi/v/text_det_metric"></a>
     <a href="https://pepy.tech/project/text_det_metric"><img src="https://static.pepy.tech/personalized-badge/text_det_metric?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads "></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
     <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

- This library is used to calculate the three metric `Precision`, `Recall` and `H-mean` to evaluate the effect of text detection algorithms. It is used in conjunction with [Modelscope-Text Detection Test Set](https://www.modelscope.cn/datasets/liekkas/text_det_test_dataset/summary).
- Indicator calculation code reference: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/b13f99607653c220ba94df2a8650edac086b0f37/ppocr/metrics/eval_det_iou.py) and [DB](https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8)

#### Overall framework
```mermaid
flowchart TD

A([Text Detect Algorithm]) --get_pred_txt.py--> B([pred_txt])
B --compute_metric.py--> C([TextDetMetric]) --> Metric(["Precision
Recall
H-mean"])
```

#### Evaluation on the dataset
- If you want to evaluate other text detection algorithms, you need to write the prediction results into `pre.txt` in the format of `image full path\t detection frame polygon coordinates\t score`
     - ⚠️Note: The full path of the image is loaded from the modelscope, as long as `txt` and `json` are in the same directory.
     - The following example:
         ```text
         C:\Users\xxxx\.cache\modelscope\hub\datasets\liekkas\text_det_test_dataset\master\data_files\extracted\f3ca4a17a478c1d798db96b03a5da8b144f13054fd06401e5a113a7ca4953491\text_det_test_data set/25.jpg [[[519.0, 634.0], [765.0, 632.0], [765.0, 683.0], [519.0, 685.0]]] [0.8451064699863124]
         ```
- Here is the evaluation code of `ch_mobile_v2_det` on the text detection test set [liekkas/text_det_test_dataset](https://www.modelscope.cn/datasets/liekkas/text_det_test_dataset/summary), and you can use the same analogy.
- Install necessary packages
     ```bash
     pip install modelscope==1.5.2
     pip install text_det_metric
     ```
- Run
     1. Run `get_pred_txt.py` to get `pred.txt`
         ```python
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
             img_path = one_data. get("image:FILE")

             print(img_path)
             img = cv2.imread(str(img_path))
             dt_boxes, scores, _ = text_detector(img)
             content.append(f"{img_path}\t{dt_boxes.tolist()}\t{scores}")

         with open("pred.txt", "w", encoding="utf-8") as f:
             for v in content:
                 f.write(f"{v}\n")
         ```
     2. Run `compute_metric.py` to get the metrics on the dataset
         ```python
         from text_det_metric import DetectionIoUEvaluator

         metric = DetectionIoUEvaluator()

         # pred_path
         pred_path = "pred.txt"
         metric = metric(pred_path)
         print(mertric)

         # {'precision': 0.6926406926406926, 'recall': 0.8247422680412371, 'hmean': 0.7529411764705882}
         ```