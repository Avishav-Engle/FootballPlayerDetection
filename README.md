# FootballPlayerDetection

### Sections:
* [Overview](#overview)
* [Installation Guide](#installation-guide)
* [Usage](#usage)
* [Files in repo](#files-in-the-repository)
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
### Prerequisites
| Library                | Version |
|------------------------|---------|
| `Python`               | `3.9`   |
| `cuda (for GPU usage)` | `11.3 ` |

### 1. Virtual Environment
#### 1.1. Create a virtual environment
```bash
python3 -m venv venv
```
#### 1.2. Activate the virtual environment
```bash
source venv/bin/activate
```
#### 1.3. Install the requirements
Install requirements and GUI for display result on sample.
```bash
pip install -r requirements.txt
```
------------------------
## Usage
```python
from PencilDrawingBySketchAndTone import *
import matplotlib.pyplot as plt
ex_img = io.imread('./inputs/11--128.jpg')
pencil_tex = './pencils/pencil1.jpg'
ex_im_pen = gen_pencil_drawing(ex_img, kernel_size=8, stroke_width=0, num_of_directions=8, smooth_kernel="gauss",
                       gradient_method=0, rgb=True, w_group=2, pencil_texture_path=pencil_tex,
                       stroke_darkness= 2,tone_darkness=1.5)
plt.rcParams['figure.figsize'] = [16,10]
plt.imshow(ex_im_pen)
plt.axis("off")
```

### Parameters
* kernel_size = size of the line segement kernel (usually 1/30 of the height/width of the original image)
* stroke_width = thickness of the strokes in the Stroke Map (0, 1, 2)
* num_of_directions = stroke directions in the Stroke Map (used for the kernels)
* smooth_kernel = how the image is smoothed (Gaussian Kernel - "gauss", Median Filter - "median")
* gradient_method = how the gradients for the Stroke Map are calculated (0 - forward gradient, 1 - Sobel)
* rgb = True if the original image has 3 channels, False if grayscale
* w_group = 3 possible weight groups (0, 1, 2) for the histogram distribution, according to the paper (brighter to darker)
* pencil_texture_path = path to the Pencil Texture Map to use (4 options in "./pencils", you can add your own)
* stroke_darkness = 1 is the same, up is darker.
* tone_darkness = as above

------------------------
## Files in repo

| File name                                                     | Purpose                                                                                                                                       |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `train_model.py`                                              | train Resnet18 model with configuration from CLIP.                                                                                            |
| `test_model.py`                                               | load trained model and test on dataset described with json annotation file (default test set of FOOD101).                                     |
| `sample_test.py`                                              | load trained model and test on single image and display with ingredients.                                                                     |
| `Data/IngredientsLoader.py`                                   | modified data loader for parsing the annotation file and the relevant images.                                                                 |


------------------------
## References
[1] Official YOLOv7 - Implementation of paper - https://github.com/WongKinYiu/yolov7

[2] Paperspace Blog "How to train and use a custom YOLOv7 model" - https://blog.paperspace.com/yolov7/
