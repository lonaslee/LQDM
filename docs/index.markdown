---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: ""
---

# Learning to Segment Liquids in Real-world Images

Jonas Li, Michelle Li, Luke Liu, Heng Fan

![Liquids](/liquid.png)

#### Abstract
---
Different types of liquids such as water, wine and medicine appear in all aspects of daily life. However, limited attention has been given to the task, hindering the ability of robots to avoid or interact with liquids safely. The segmentation of liquids is difficult because liquids come in diverse appearances and shapes; moreover, they can be both transparent or reflective, taking on arbitrary objects and scenes from the background or surroundings. To take on this challenge, we construct a large-scale dataset of liquids named LQDS consisting of 5000 real-world images annotated into 14 distinct classes, and design a novel liquid detection model named LQDM, which leverages cross-attention between a dedicated boundary branch and the main segmentation branch to enhance segmentation predictions. Extensive experiments demonstrate the effectiveness of LQDM on the test set of LQDS, outperforming state-of-the-art methods and establishing a strong baseline for the semantic segmentation of liquids.

[[Paper](https://arxiv.org/abs/2601.00940)]

<!-- - [Code](https://github.com/lonaslee/LQDM) -->
<!-- - [Benchmark](https://drive.google.com/drive/folders/1KIouI3V6XdIfXIgko9MYjBQfar183Wej?usp=sharing) -->
<!-- - [Supplementary](supplementary.pdf) -->

#### Benchmark
---
![Benchmark](/benchmark.png)

LQDS contains 5K images of liquids, each with corresponding liquid masks. The liquid
objects in the images are categorized into 14 classes: water, wine, juice, cocktails, 
soda, coffee, tea, boba, chemical, medical, milk, spirits, honey, and miscellaneous.

<!-- [[Download](https://lonaslee.github.io/LQDM)] -->

#### Method
---

![Arch](/arch.png)

The proposed LQDM is a dual-branch architecture which injects boundary features into 
segmentation mask predictions through cross-attention, setting the benchmark for the
liquid segmentation task with a 59.28% mean IoU and a 71.61% mean pixel-accuracy.
