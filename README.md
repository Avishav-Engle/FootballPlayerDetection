# FootballPlayerDetection

### Sections:
* [Overview](#overview)
* [Installation Guide](#installation-guide)
* [Usage](#usage)
* [Files in repo](#files-in-repo)
* [References](#references)
------------------------
## Overview
In this project we used fine-tuning of a YOLOv7 network and implementation and training of a BYOL network to classify, localise and split into teams, players and other objects in football games.

![image](https://user-images.githubusercontent.com/82023333/214358132-5ebc7b7d-0fbf-4df0-8e43-23e68fc774b9.png)

Draws inspiration from a similar basketball-based implementation by James Skelton - https://blog.paperspace.com/yolov7/

In this notebook, we will explain and implement the algorithm.


We can divide the workflow into 2 main steps:
1. Localisation and Classification of football match **image** or **video**.
2. Clustering of players from **image** to their two teams.

Combining the results from these steps should yield the desired result. The workflow can be depicted as follows:
![image](https://user-images.githubusercontent.com/82023333/214572202-50d54131-742e-4efa-b7d7-7fb001589364.png)

------------------------
## Installation Guide
### Prerequisites (flexible)
| Library                | Version |
|------------------------|---------|
| `Python`               | `3.8`   |
| `cuda (for GPU usage)` | `11.6` |

### 1. Virtual Environment
(for example)
#### 1.1. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Download the weights
Download the weights for each of the networks, they're stored here:
https://drive.google.com/drive/folders/1rpshi_ZycbRSJtwy1QXx73oTUWTzcJgk?usp=sharing

### 4. Clone this repo

------------------------
## Usage
Open a terminal inside the cloned repo and run the following:
```bash
python run.py --yolo-weights 'path/to/YOLO/weights.pt' --byol-weights 'path/to/BYOL/weights.pt' --source 'path/to/source/image_or_video'
```

### Parameters
* yolo-weights = pretrained weights for YOLO network
* byol-weights = pretrained weights for BYOL network
* source = path to source image or video
* img-size = inference size in pixels (default=640)
* conf-thres = object confidence threshold (default=0.25)
* iou-thres = IOU threshold for NMS (default=0.45)
* device = cuda device, i.e. 0 or 0,1,2,3 or cpu (default='')
* view-img = display results
* save-txt = save results to *.txt
* save-conf = save confidences in --save-txt labels
* nosave = do not save images/videos
* classes = filter by class: --class 0, or --class 0 2 3 (0:ball, 1:goalkeeper, 2:player, 3:referee)
* agnostic-nms = class-agnostic NMS
* augment = augmented inference
* project = save results to project/name (default='detections')
* name = save results to project/name (default='run')
* exist-ok = existing project/name ok, do not increment
* no-trace = don`t trace model

------------------------
## Files in repo

| File name                 | Purpose                                                                            |
|---------------------------|------------------------------------------------------------------------------------|
| `Readme.md`               | Explanation of repo contents and how to use them                                   |
| `requirements.txt`        | Requirements file for all the algorithms                                           |
| `yolov7-main`             | Directory of YOLOv7 codebase                                                       |
| `byol-main.py`            | BYOL codebase                                                                      |
| `run.py`                  | Run end to end detect algorithm.                                                   |
| `Other`                   | Directory with all other code we used during training                              |



------------------------
## References
[1] Official YOLOv7 - Implementation of paper - https://github.com/WongKinYiu/yolov7

[2] Paperspace Blog "How to train and use a custom YOLOv7 model" - https://blog.paperspace.com/yolov7/

[3] BYOL - Technion DL Tutorial - Tal Daniel - https://github.com/taldatech/ee046211-deep-learning/blob/main/ee046211_tutorial_09_self_supervised_representation_learning.ipynb
