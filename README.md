# HeadPoseEstimate
## Introduce
This is a head pose estimation system based on 3d facial landmarks. Please realize it's not the most advanced method in this field. Util I created this repository, there have been some end-to-end solutions.

## Usage
1. Download https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar to `model` and rename it as `3DFAN4.pth.tar`.
2. Download https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar to `model` and rename it as `depth.pth.tar`.
3. Run `python estimate_head_pose.py -i <path of image>`. The webbrowser will be opened to show 3d landmarks.


## How does it work
### 1. Get the 3d facial landmarks
Thanks for 1adrianb's [excellent work](https://github.com/1adrianb/face-alignment), we can easily get the 3d facial landmarks.
### 2. Determine direction of face
The horizontal direction `hd` and vertical direction `vd` of face can be determined by PCA. Let's notate facial orientation with `fd`, then `fd = hd x vd`. Note: `x` is cross products.  
Here is an example. The origin image:
![origin image](figures/origin_image.png)

The following image shows 68 landmarks.  
Red axis: X  
Green axis: Y  
Blue axis: Z  
The three yellow arrows are `hd`, `vd` and `fd`.
![landmarks](figures/landmarks.png)
### 3. Estimate rotation
Normalize `hd`, `vd` and `fd`, make them as unit vectors. 
Rotation transform from  
   | 1  0  0 |  
   | 0  1  0 |  
   | 0  0  1 |  
   to  
   | hd |  
   | vd |  
   | fd |  
can be estimated with Kabsch algorithm.

## Motivation behind this solution
There are several pose estimation algorithms. They can be broadly separated into three categories:  
* End-to-End solution: Most of them are based on dnn, you can get the pose directly from a single image without facial landmarks or internal parameter matrix. I highly recommend you to take a look on [deep-head-pose](https://github.com/natanielruiz/deep-head-pose).
* Perspective-n-Points: This is the problem of estimating the pose of a calibrated camera given a set of n 3D points in the world and their corresponding 2D projections in the image. It's based on some geometric theories which are clear and definite.
However, the drawbacks are also obvious: first, if you require high precision, a 3d model is necessary, which are not available in most cases; second, camera model and internal parameter matrix(they can be roughly estimated) are also not available in most cases; third, sometime we can't get exact solution of a PnP problem.
The overlapped error can't be ignored.
* Kabsch Algorithm: It is a method for calculating the optimal rotation matrix that minimizes the RMSD (root mean squared deviation) between two paired sets of 3D points. Just like PnP solver we talked above, Kabsch Algorithms is also supported by mathematical principles. Compared with PnP solver, Kabsch Algorithm gets rid of camera model, internal parameter matrix and make itself more practical in real scenes. Thanks to the [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment), 3d facial landmarks are available currently and the pipeline is composed of 3 steps described in `How does it work`.

In the end, let me point out the shortcomings of our mothod:
1. The result depend on 3d facial landmarks. High error we'll get if the 3d facial landmarks are not accurate.
2. It's not real-time. PCA, SVD are involved.