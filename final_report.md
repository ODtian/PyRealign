# TItle 

Luyue Tian,Xiang Li,Chenyang Mao,Zining Song,Bingshen Qian,Tian Pu

lixiang2022@shanghaitech.edu.cn

tianly2022@shanghaitech.edu.cn
maochy2022@shanghaitech.edu.cn

## Abstract
We propose a JAX-powered framework for 3D B-spline interpolation and rigid-body registration of 4D neuroimaging data. Leveraging JAX's automatic differentiation and GPU acceleration, the system enables efficient, differentiable motion correction across temporal frames. The BSpline module supports arbitrary-degree spline interpolation, while the Realign module estimates transformations iteratively. An interactive notebook provides visualization of voxel grids and alignment results, making the pipeline suitable for large-scale fMRI analysis.

## Keywards
JAX,GPU acceleration,Visualization,Image registration

## Metadata
|Nr |Code metadata description |Metadata |
|----|----------------------------|----------|
|C1 |Current code version   |   latest|
|C2 | Permanent link to code/repository used for this code version   |       https://github.com/ODtian/PyRealign/blob/main/requirements.txt            |
|C3 | ermanent link to reproducible capsule |                   |
|C4 | Legal code license | MIT license |
|C5 | Code versioning system used | git |
|C6 | Software code languages, tools and services used | python |
|C7 | Compilation requirements, operating environments and dependencies |                   |
|C8 | If available, link to developer documentation/manual |                   |
|C9 | Support email for questions |                   |

## Motivation and significance
In the field of neuroimaging, image registration is crucial for aligning temporal or multimodal brain data to facilitate accurate analysis. The original implementation in final.py provides a classic rigid-body registration framework using B-spline interpolation, but it suffers from high computational overhead, limited scalability, and non-differentiability, making it unsuitable for modern high-throughput workflows or integration with learning-based pipelines.

To address these limitations, we reimplemented the framework in bspline.py using JAX, enabling automatic differentiation, just-in-time (JIT) compilation, and GPU acceleration. The new implementation not only retains the core principles of B-spline interpolation but also introduces differentiable optimization, vectorized coordinate sampling, and batch registration using vmap, significantly improving speed and flexibility.

This software is designed to solve the scientific problem of large-scale, differentiable motion correction in 4D fMRI datasets. It supports voxel-level alignment with sub-voxel accuracy and enables easy integration into machine learning pipelines. The user can load 4D neuroimaging data, configure interpolation degree and optimization parameters, and apply registration across timepoints using a single call.

Related work includes traditional registration toolkits like ANTs, SPM, and FSL, as well as recent learning-based frameworks such as VoxelMorph. However, most of these either lack differentiability or require large amounts of training data. Our method balances physical interpretability, computational efficiency, and differentiable capability, making it suitable for hybrid learning-registration pipelines or scientific discovery in brain dynamics.

## 2.Software Description
This software implements a high-performance 3D B-spline interpolation framework for neuroimaging data registration and transformation. Written in Python and powered by JAX for GPU acceleration, it provides tools for smooth, differentiable spatial transformations on 3D volumes, supporting operations such as realignment, sampling, and interpolation using B-spline basis functions of arbitrary degree.
### Software Architecture
- Interpolation Core (BSpline class)
    - Implements N-dimensional B-spline interpolation.

    - Dynamically generates knot vectors and padding based on input data and interpolation mode (reflect, clamp, or none).

    - Supports efficient evaluation using JAX jit and vmap to accelerate computation on CPUs or GPUs.
- Registration Module (Realign class)
    - Reads 4D neuroimaging data using ITK, converts to JAX arrays;
    - Performs rigid transformation estimation via a differentiable loss;
    - Constructs voxel-to-cartesian coordinate transform matrices, samples 3D grid points, and optimizes transformation parameters using Gauss-Newton-style updates.
- Execution & Visualization (main/show)
    - The main() function runs registration for a batch of images using vmap;
    - The show() function provides 2D visualization for volumetric slices using matplotlib.
  

### 2.2 Software Functionalities


## 3 Illustrative examples
We demonstrate the functionality of our JAX-based B-spline registration framework using a publicly available 4D fMRI dataset. The dataset contains a series of temporally ordered brain volumes, each of which may be affected by motion artifacts. The goal of the experiment is to align each volume to a fixed reference frame using rigid-body transformation and differentiable optimization.
Step 1: Load and Initialize

The user loads a .nii.gz file using ITK. The software automatically extracts voxel spacing and initializes transformation matrices. A reference image is selected (typically the first frame).
Step 2: Run Optimization

Using Realign.estimate(index), rigid-body parameters for each volume are optimized iteratively via Gauss–Newton-like updates. JAX's autodiff and jit compilation significantly reduce runtime.
Step 3: Visualize Parameters and Warp

The estimated parameters (translations and rotations) are visualized as motion curves. Optionally, transformed coordinates can be rendered, and aligned volumes exported using Nibabel. The included show() function allows visualization of 2D slices before and after alignment.
Output

    Registration parameter plot (translation/rotation per frame)

    Visualization of original vs. aligned slices

    Registered .nii.gz output file


## 4 Impact
This software was developed as part of a course project, with the main goal of accelerating the B-spline-based image registration process commonly used in neuroimaging tasks. The original implementation relied on CPU-based iterative loops with minimal optimization, which made it time-consuming and difficult to scale to larger 4D datasets.

By re-implementing the algorithm using JAX, we were able to take advantage of just-in-time (JIT) compilation and GPU acceleration. As a result, the new version achieves significantly faster execution while maintaining comparable registration accuracy. In particular, functions such as B-spline evaluation, Jacobian computation, and coordinate transformation are now fully vectorized and optimized.

## 5 Conclusion
We present a JAX-based implementation of B-spline interpolation and rigid-body image registration aimed at improving computational efficiency in neuroimaging tasks. Compared to the original CPU-based version, our redesigned framework takes advantage of JAX’s automatic differentiation, vectorization, and GPU acceleration, enabling faster execution while maintaining algorithmic transparency and accuracy. Although developed as a course project, the software demonstrates how modern numerical tools can enhance classical medical image processing workflows and may serve as a basis for further research or teaching use in motion correction and fMRI preprocessing.