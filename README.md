# FootballPlayerDetection

In this project we used fine-tuning of a YOLOv7 network and implementation and training of a BYOL network to classify, localise and split into teams, players and other objects in football games.

![image](https://user-images.githubusercontent.com/82023333/214358132-5ebc7b7d-0fbf-4df0-8e43-23e68fc774b9.png)

Draws inspiration from a similar basketball-based implementation by James Skelton - https://blog.paperspace.com/yolov7/

In this notebook, we will explain and implement the algorithm.

We can divide the workflow into 2 main steps:
1. Localisation and Classification of football match **image** or **video**.
2. Clustering of players from **image** to their two teams.

Combining the results from these steps should yield the desired result. The workflow can be depicted as follows:
![image](https://user-images.githubusercontent.com/82023333/214570149-78c8ac6d-2725-493b-a1fc-227fd09cc1af.png)

# Usage
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
# Parameters
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

# Folders
* inputs: test images from the publishers' website: http://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/pencil_drawing.htm
* pencils: pencil textures for generating the Pencil Texture Map

# Reference
[1] Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]//Proceedings of the Symposium on Non-Photorealistic Animation and Rendering. Eurographics Association, 2012: 65-73.

[2] Paperspace Blog "How to train and use a custom YOLOv7 model" - https://blog.paperspace.com/yolov7/
