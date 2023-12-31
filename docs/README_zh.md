[English](https://github.com/SWHL/TextDetMetric) | 简体中文

## Text Detect Metric
<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python->=3.6,<3.12-aff.svg"></a>
    <a href="https://pypi.org/project/text_det_metric/"><img alt="PyPI" src="https://img.shields.io/pypi/v/text_det_metric"></a>
    <a href="https://pepy.tech/project/text-det-metric"><img src="https://static.pepy.tech/personalized-badge/text-det-metric?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

- 该库用于计算`Precision`、`Recall`和`H-mean`三个指标，用来评测文本检测算法效果。与[魔搭-文本检测测试集](https://www.modelscope.cn/datasets/liekkas/text_det_test_dataset/summary)配套使用。
- 指标计算代码参考：[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/b13f99607653c220ba94df2a8650edac086b0f37/ppocr/metrics/eval_det_iou.py) 和 [DB](https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8)

#### 整体框架
```mermaid
flowchart LR

A([Text Detect Algorithm]) --get_pred_txt.py--> B([pred_txt])
B --compute_metric.py--> C([TextDetMetric]) --> D([Precision])
C --> E([Recall])
C --> F([H-mean])
```

#### 自己数据集上评测
- 如果想要评测其他文本检测算法，需要将预测结果写入`pre.txt`中，格式为`图像全路径\t检测框多边形坐标\t耗时`
- ⚠️注意：图形全路径来自modelscope加载得到，只要保证`txt`和`json`在同一目录下即可。
- 如下示例：
    ```text
    C:\Users\xxxx\.cache\modelscope\hub\datasets\liekkas\text_det_test_dataset\master\data_files\extracted\f3ca4a17a478c1d798db96b03a5da8b144f13054fd06401e5a113a7ca4953491\text_det_test_dataset/25.jpg	[[[519.0, 634.0], [765.0, 632.0], [765.0, 683.0], [519.0, 685.0]]]	0.2804088592529297
    ```

- 这里以`ch_mobile_v2_det`在文本检测测试集[liekkas/text_det_test_dataset](https://www.modelscope.cn/datasets/liekkas/text_det_test_dataset/summary)上的评测代码，大家可以以此类推。
- 安装必要的包
    ```bash
    pip install modelscope==1.5.2
    pip install text_det_metric
    ```
- 运行测试
    1. 运行`get_pred_txt.py`得到`pred.txt`
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
    2. 运行`compute_metric.py`得到在该数据集上的指标
        ```python
        from text_det_metric import DetectionIoUEvaluator

        metric = DetectionIoUEvaluator()

        # pred_path
        pred_path = "pred.txt"
        metric = metric(pred_path)
        print(metric)

        # {'precision': 0.6926406926406926, 'recall': 0.8247422680412371, 'hmean': 0.7529411764705882}
        ```
