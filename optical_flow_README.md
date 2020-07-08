Optical flow from two images
================================
Kenta Tanaka & Eiji Watanabe modified by Lana S.


Overview
================================
Using two images as input, it outputs the result image of the optical flow and the calculated vector group as csv file.
You can choose Lucas-Kanade method or Gunnar-Farneback method.



Test environment
===========================
OS: Ubuntu 14.04 or 16.04
Python: 2.7



Requirements
===========================
* cv2
* yaml
* itertools
* csv
* argparse

## Install

`pip install -U PyYAML`



How to use
================================
To calculate by the Lucas-Kanade method, execute as follows.

  $ python optical_flow.py penguin1.png penguin2.png

To calculate by the Gunnar-Farneback method, execute as follows.

  $ python optical_flow.py penguin1.png penguin2.png -m fb

An image in which only the vector of the optical flow is drawn is output as vectors.png,
 and an image overlaid with the vector on penguin2.png is output to result.png.

In addition, data that arranges the start point of the vector and the direction of the vector are output to data.csv.
If the starting point of the vector is (px, py) and the direction of the vector is (dx, dy),
the format of data.csv is as follows.

px1, py2, dx1, dy2
px1, py2, dx1, dy2
...


Sample commands for snake illusion:
 $ python optical_flow.py test_20y_0.jpg test_20y_1.jpg -cc yellow -lc red -s 2 -l 2 -vs 60.0
 $ python optical_flow.py test_20y_1.jpg test_20y_2.jpg -cc yellow -lc red -s 2 -l 2 -vs 60.0
 $ python optical_flow.py test_20y_0.jpg test_20y_1.jpg -cc yellow -lc red -s 2 -l 2 -vs 90.0
 $ python optical_flow.py test_20y_1.jpg test_20y_2.jpg -cc yellow -lc red -s 2 -l 2 -vs 90.0
 $ python optical_flow.py test_20y_0.jpg test_20y_1.jpg -cc yellow -lc red -s 2 -l 2 -vs 180.0
 $ python optical_flow.py test_20y_1.jpg test_20y_2.jpg -cc yellow -lc red -s 2 -l 2 -vs 180.0




Options
================================
parser.add_argument('image1', type=str, help='Input image No1.')
parser.add_argument('image2', type=str, help='Input image No2.')
parser.add_argument('--method', '-m', type=str, default='lk', choices=['lk', 'fb'], help='Select a method.')
parser.add_argument('--circle_color', '-cc', type=str, default='red', choices=colormap.keys(), help='Select a color for circle.')
parser.add_argument('--line_color', '-lc', type=str, default='red', choices=colormap.keys(), help='Select a color for line.')
parser.add_argument('--vector_scale', '-vs', type=float, default=1.0, help='Scale saving vector data.')
parser.add_argument('--size', '-s', type=int, default=5, help='Size of original point marker.')
parser.add_argument('--line', '-l', type=int, default=2, help='Width of vector line.')



Parameter setting of Lucas-Kanade
================================
To adjust the parameters of the Lucas-Kanade method, change the parameters after LucasKanade in config.yaml.

** window_size **
Specify the window size when calculating the optical flow.
If this value is small, a more local velocity vector is calculated,
and if this value is large, a more global velocity vector is calculated.

** quality_level **
Sets the quality of feature detection.
The smaller the value, the smaller the threshold value as the feature value,
and it will extract many feature points.



Parameter setting of Gunnar-Farneback
================================
To adjust the parameters of Gunnar-Farneback method, change the parameter after Farneback in config.yaml.

** window_size **
Specify the window size when calculating the optical flow.
If this value is small, a more local velocity vector is calculated,
and if this value is large, a more global velocity vector is calculated.

** stride **

In the Gunnar-Farneback method, since velocity vectors are calculated for all pixels,
data is thinned out by specifying stride.
"stride: 1" means no thinning, "stride: 5" data is saved every 5 pixels.

** min_vec **
Sets the minimum value of the vector length to be saved.
Vectors less than or equal to the length specified here will not be saved or drawn.



Reference
===========================
"http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kan".
"https://doi.org/10.3389/fpsyg.2018.00345"

"https://doi.org/10.3389/fpsyg.2018.00345" [Paper]
Watanabe E, Kitaoka A, Sakamoto K, Yasugi M and Tanaka K (2018)
Illusory Motion Reproduced by Deep Neural Networks Trained for Prediction.
Front. Psychol. 9:345. doi: 10.3389/fpsyg.2018.00345

