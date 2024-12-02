Optisplat
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Rahul Aggarwal, Josh Smith, Mile Rabbitz
  * Linkedins [TODO] ...
* Tested on: Ubuntu 20.04 x86_64, AMD EPYC 7452 32-Core (128) @ 2.35GHz 512GB, NVIDIA RTX A6000 48GB

## Overview (What is optimal splatting?)

This project addresses the inherent limitations of traditional gaussian splatting techniques in rendering 3D scenes. Standard gaussian splatting involves approximating the projection of 3D gaussians into 2D splats using a first order approximation as the perspective transformation for 3D gaussians does not have a closed form. These projection errors degrade the quality of both training and inference in Gaussian splatting-based rendering.

A recent work [1] has proposed an alternative approach to improve the fidelity of Gaussian projections under perspective transformations. Instead of projecting Gaussians onto a flat plane, they propose radially projecting each 3D gaussian based on its mean along a line that connects it directly to the camera center. The projection lands each Gaussian on a plane tangent to the unit sphere and perpendicular to the projection line.

![optisplat_pipeline](assets/docs/rendering/output_video.gif)

## Objectives (What are we aiming to achieve?)

We aim to make the following key contributions:

- [x] **Optimize Optimal Gaussian Splatting**. Currently, the baseline performance of optimal gaussian splatting when compared to the original implementation is a 3x *slowdown.* We aim to improve performance on optimal gaussian splatting by 66% by writing custom CUDA parallel code for both the forward and backward passes.

- [ ] **Write the novel fisheye camera** forward and backward passes, making it possible to train and render gaussian splats directly from fisheye lenses. This will involve deriving the gradient between the local tangent plane and the image plane projections.

- [x] **Provide the first 360 degree camera gaussian splatting SLAM implementation**. Gaussian splatting SLAM has been done before with narrow FOV, pinhole cameras. However, 360 degree cameras offer a major challenge as they have a much wider FOV, which causes traditional gaussian splatting SLAM implementations to fail. Additionally, we have to deal with a different camera model, which we aim to first solve through rectification, and then through our fisheye implementation.


## Optisplat Installation

Clone the repository and create an anaconda environment using

```shell
git clone https://github.com/rahulaggarwal965/cis5650-final-project.git
cd cis5650-final-project

SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate optisplat
```

This repository contains support for various camera models: To install them, pick the specified camera model and run the command below. **We recommend starting with the pinhole implementation**.

```shell
# Pinhole
pip install submodules/diff-gaussian-rasterization-pinhole-opt
```

```shell
# Panorama
pip install submodules/diff-gaussian-rasterization-panorama
```

```shell
# Fisheye
pip install submodules/diff-gaussian-rasterization-fisheye-opt
```

## Datasets

### Mip-NeRF 360 Dataset

Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/).

### Tanks & Temples dataset

Please download the data from the [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

### Deep Blending

Please download the data from the [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

## Training and Evaluation

By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval # Train with train/test split
python render.py -m <path to trained model> --fov_ratio 1 # Generate renderings
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --fov_ratio
  Focal length reduction ratios.

</details>

## OptiSplat Optimizations

One of our main goals of this project was to make optimal splatting fast enough for it to be feasible to use in a real-time SLAM system. To that end, here are a few optimizations we made. We give a full performance comparison in the summary section below. Additionally, note that forward-pass optimizations affect the backward-pass, but not vice-versa.

### Forward

#### Separated Spherical Harmonics

In the original 3DGS implementation, both the base color (0th SH band) and higher SH bands were merged into a single tensor prior to rasterization, leading to noticeable performance hits during the forward pass. To address this, we've restructured the differential rasterizer to load the Gaussian SH coefficients from separate tensors, streamlining the process and boosting performance.

Since the base color has a significantly higher weight in the color computation, combining it with the other SH bands introduces unnecessary computational overhead.

#### Tangent Plane Preloading

When computing the tangent plane of each gaussian on the unit sphere, we have to do an expensive calculation involving projecting the ray, caculating the spherical coordinates for the ray and assigning a local coordinate frame to the resulting tangent plane. We note that the parallelization scheme for gaussian splatting assigns each *tile* a CUDA block and each *pixel* within a tile a thread. However, this means that each pixel must iterate through each Gaussian, computing the expensive ray intersection each time. Therefore, we implement a caching mechanism that stores the $sin$ and $cos$ of the spherical coordinates (unit length) in shared memory, loaded a warp at a time. This both reduces memory bandwith and computational overhead, which greatly improves performance.

![tangent_plane_preloading](assets/docs/tangent_plane_preloading.png)

### Backward

#### Sparse Gradient Propagation

During training, we know that each camera can only see a part of the scene. Therefore, we would not like to optimize the paremeters of all the gaussians in the scene. To do so, we may use a [Sparse ADAM](https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html) optimizer. and a mask based on whether a Gaussian radius is on-screen or not. However, because of the nature of optimal splatting, gaussians not in view can still have a contribution to the resulting image due to their *tangent-plane contribution*. Therefore, we take this into account when calculating sparsity.

![sparse_gradient_prop](assets/docs/sparse_opt/output_video.gif)

#### Fused SSIM

The Fused SSIM optimization was taken from [2], and involves optimizing the structural similarity index metric, which is used during loss computation. Below, we provide an excerpt from the paper.

>  3DGS loss computation involves evaluating the SSIM metric. It is configured to use 11×11 Gaussian kernel convolu-tion: we propose using optimized CUDA kernels to perform differentiable 2D convolution via two consecutive 1Dconvolutions since Gaussian kernels are separable in nature. In addition, we use a fused kernel for the evaluation of theSSIM metric from the convolved results. This speeds up the loss calculation and is particularly impactful when thenumber of optimized Gaussians is low compared to image resolution, which is the case when training on a budget.

#### Warp-level Gradient Accumulation

TODO (mike):

### Performance Summary

Our performance metrics were tested on a number of scenes, and overall we see a 37% average improvement due to our various optimizations. Below, we provide an analysis of the Bonsai scene to give a specific example of our optimization's performance on a medium size scene. **Note that the following scene is randomly picked**. We display example renderings and performance summar  below.

|fov_ratio|GT|Render|
|--|--|--|
|1|![gt](assets/renders/gt.png)|![fov_ratio_0_1](assets/renders/fov_ratio_1_pinhole_render.png)|
|0.1|![gt](assets/renders/gt.png)|![fov_ratio_0_1](assets/renders/fov_ratio_0_1_pinhole_render.png)|

Above, we see that our implementation performs quite well even as we increase the FOV, which thereby increases the number of Gaussians that need to be rendered.

![perf_summary](assets/docs/perf_summary.png)

Overall, we see a 37% improvement on the optimization phase of gaussian splatting, especially as we optimize further. We note that the longer the optimization process goes, the number of gaussians increase. However, we see that we also increase in our performance gain, implying that our optimizations perform better for more expensive operations.

We see that the most important optimization is the separating of the spherical harmonic components, which makes sense because it applies to both the forward and backward passes. The Sparse ADAM, Fused SSIM, and tangent preloading all provide smaller (~5%) improvements that add to solid performance.

## Fisheye Camera Model Implementation



## Gaussian Splatting SLAM


## Gaussian Splatting SLAM Installation

To install our custom compatible Gaussian Splatting SLAM installation, clone the repository using
```shell
git submodule init
git submodule update
```

Verify that in `submodules/MonoGS`, you have checked out the `optisplat.speedup` branch. Then, install the necessary requirements:

```shell
pip install -r requirements.txt
```

Note: Please see [lycon](https://github.com/ethereon/lycon) and [lietorch](https://github.com/princeton-vl/lietorch) to ensure that the proper dependencies are installed so that the speedup may be enabled.

Finally, to enable support for pose optimization, run the command below:

```shell
# Pinhole with pose optimization
pip install submodules/diff-gaussian-rasterization-pinhole-opt-w-pose
```


## Acknowledgements

This project is built upon [op43dgs](https://github.com/LetianHuang/op43dgs). Please follow the license of op43dgs. We thank all the authors for their great work. Additonally, we build heavily upon [MonoGS](https://github.com/muskie82/MonoGS). 

## Documentation

* [Project Pitch](https://docs.google.com/presentation/d/12cumM1NYkJ7Nz2546gtauragFJXBte2i5zr2OOwhN4w/edit?usp=sharing)  
* [Milestone 1](https://docs.google.com/presentation/d/1kTaBZR_AYzOxLMjsLN8go0clONUSYL47gi0aoIQo9Wg/edit?usp=sharing)  
* [Milestone 2](https://docs.google.com/presentation/d/1asgsxSQ5VbEdhZiO-TxcCMCsjYq1s7_qwXEn_w4wN_g/edit?usp=drive_link)
* [Milestone 3](https://docs.google.com/presentation/d/1g-BpCL5niX12zsmYnQpC8C8AzdGHj43fHt41Tv6y5xk/edit?usp=drive_link)

## References

- [1] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy: https://arxiv.org/pdf/2402.00752

- [2] Taming 3DGS: High-Quality Radiance Fields with Limited Resources: https://arxiv.org/pdf/2406.15643