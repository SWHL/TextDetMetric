## Text Detect Metric
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python->=3.6,<3.12-aff.svg"></a>
    <a href="https://pypi.org/project/text_det_metric/"><img alt="PyPI" src="https://img.shields.io/pypi/v/text_det_metric"></a>
    <a href="https://pepy.tech/project/text_det_metric"><img src="https://static.pepy.tech/personalized-badge/text_det_metric?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


### Evaluate on the custom dataset.
- Here we use the evaluation code of `ch_mobile_v2_det` on the text detection test set [liekkas/text_det_test_dataset](https://www.modelscope.cn/datasets/liekkas/text_det_test_dataset/summary), and you can use the same analogy.


#### 1. Install packages.
```bash
pip install modelscope==1.5.2
pip install text_det_metric
```

#### 2. Run
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
        img_path = one_data.get("image:FILE")

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
    pred_path = "1.txt"
    mertric = metric(pred_path)
    print(mertric)
    ```

### See details for [TextDetMetric](https://github.com/SWHL/TextDetMetric).
