Assignment 3
===================================

# A. Theory 
## Transmittance Calculation (10 points)
Transmittance calculation is a core part for the implementation of volume rendering. Your first task is to compute the transmittance of a ray going through a non-homogeneous medium (shown in the image below).   
Please compute the transmittace in `transmittance_calculation/a3_transmittance.pdf` and submit the result along with your code for part B to canvas. You can either hand write the result or edit the tex file, as long as it is readable by the TAs.

![Transmittance computation](transmittance_calculation/figure1.png)


# B. Implementation

##  0. Setup

### 0.1 Environment setup
You can use the python environment you've set up for past assignments, or re-install it with our `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate l3d
```

If you do not have Anaconda, you can quickly download it [here](https://docs.conda.io/en/latest/miniconda.html), or via the command line in with:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 0.2 Data

The data for this assignment is provided in the github repo under `data/`. You do not need to download anything yourself.

##  1. Differentiable Volume Rendering

In the emission-absorption (EA) model that we covered in class, volumes are described by their *appearance* (e.g. emission) and *geometry* (absorption) at *every point* in 3D space. For part 1 of the assignment, you will implement a ***Differentiable Renderer*** for EA volumes, which you will use in parts 2 and 3. Differentiable renderers are extremely useful for 3D learning problems --- one reason is because they allow you to optimize scene parameters (i.e. perform inverse rendering) from image supervision only!

##  1.1. Familiarize yourself with the code structure

There are four major components of our differentiable volume rendering pipeline:

* ***The camera***: `pytorch3d.CameraBase`
* ***The scene***: `SDFVolume` in `implicit.py`
* ***The sampling routine***: `StratifiedSampler` in `sampler.py`
* ***The renderer***: `VolumeRenderer` in `renderer.py`

`StratifiedSampler` provides a method for sampling multiple points along a ray traveling through the scene (also known as *raymarching*). Together, a sampler and a renderer describe a rendering pipeline. Like traditional graphics pipelines, this rendering procedure is independent of the scene and camera.

The scene, sampler, and renderer are all packaged together under the `Model` class in `main.py`. In particular the `Model`'s forward method invokes a `VolumeRenderer` instance with a sampling strategy and volume as input.

Also, take a look at the `RayBundle` class in `ray_utils.py`, which provides a convenient wrapper around several per-ray inputs to the volume rendering procedure.

##  1.2. Outline of tasks

In order to perform rendering, you will implement the following routines:

1. **Ray sampling from cameras**: you will fill out methods in `ray_utils.py` to generate world space rays from a particular camera.
2. **Point sampling along rays**: you will fill out the `StratifiedSampler` class to generate sample points along each world space ray
3. **Rendering**: you will fill out the `VolumeRenderer` class to *evaluate* a volume function at each sample point along a ray, and aggregate these evaluations to perform rendering.

##  1.3. Ray sampling (10 points)

Take a look at the `render_images` function in `main.py`. It loops through a set of cameras, generates rays for each pixel on a camera, and renders these rays using a `Model` instance.

### Implementation

Your first task is to implement:

1. `get_pixels_from_image` in `ray_utils.py` and
2. `get_rays_from_pixels` in `ray_utils.py`

which are used in `render_images`:

```python
xy_grid = get_pixels_from_image(image_size, camera) # TODO: implement in ray_utils.py
ray_bundle = get_rays_from_pixels(xy_grid, camera) # TODO: implement in ray_utils.py
```

The `get_pixels_from_image` method generates pixel coordinates, ranging from `[-1, 1]` for each pixel in an image. The `get_rays_from_pixels` method generates rays for each pixel, by mapping from a camera's *Normalized Device Coordinate (NDC) Space* into world space.

### Visualization

You can run the code for part 1 with:

```bash
python main.py --config-name=box
```

Once you have implemented these methods, verify that your output matches the TA output by visualizing both `xy_grid` and `rays` with the `vis_grid` and `vis_rays` functions in the `render_images` function in `main.py`. **By default, the above command will crash and return an error**. However, it should reach your visualization code before it does. The outputs of grid/ray visualization should look like this:

![Grid](ta_images/grid.png)    ![Rays](ta_images/rays.png)

##  1.4. Point sampling (10 points)

### Implementation

Your next task is to fill out `StratifiedSampler` in `sampler.py`. Implement the forward method, which:

1. Generates a set of distances between `near` and `far` and
2. Uses these distances to sample points offset from ray origins (`RayBundle.origins`) along ray directions (`RayBundle.directions`).
3. Stores the distances and sample points in `RayBundle.sample_points` and `RayBundle.sample_lengths`

### Visualization

Once you have done this, use the `render_points` method in `render_functions.py` in order to visualize the point samples from the first camera. They should look like this:

![Sample points](ta_images/sample_points.png)

##  1.5. Volume rendering (30 points)

Finally, we can implement volume rendering! With the `configs/box.yaml` configuration, we provide you with an `SDFVolume` instance describing a box. You can check out the code for this function in `implicit.py`, which converts a signed distance function into a volume. If you want, you can even implement your own `SDFVolume` classes by creating new signed distance function class, and adding it to `sdf_dict` in `implicit.py`. Take a look at [this great web page](https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm) for formulas for some simple/complex SDFs.

### Implementation

You will implement

1. `VolumeRenderer._compute_weights` and
2. `VolumeRenderer._aggregate`.
3. You will also modify the `VolumeRenderer.forward` method to render a depth map in addition to color from a volume

From each volume evaluation you will get both volume density, and a color (`feature`):

```python
# Call implicit function with sample points
implicit_output = implicit_fn(cur_ray_bundle)
density = implicit_output['density']
feature = implicit_output['feature']
```

You'll then use the following equation to render color along a ray:

![Spiral Rendering of Part 1](ta_images/color.png)

where `σ` is density, `Δt` is the length of current ray segment, and `L_e` is color:

![Spiral Rendering of Part 1](ta_images/transmittance.png)

Compute the weights `T * (1 - exp(-σ * Δt))` in `VolumeRenderer._compute_weights`, and perform the summation in `VolumeRenderer._aggregate`. Note that for the first segment `T = 1`.

Use weights, and aggregation function to render *color* and *depth* (stored in `RayBundle.sample_lengths`). 

### Visualization

By default, your results will be written out to `images/part_1.gif`. Provide a visualization of the depth in your write-up.

![Spiral Rendering of Part 1](ta_images/part_1.gif) ![Spiral Rendering of Part 1](ta_images/depth.png)


##  2. Optimizing a basic implicit volume

##  2.1. Random ray sampling (5 points)

Since you have now implemented a differentiable volume renderer, we can use it to optimize the parameters of a volume! We have provided a basic training loop in the `train` method in `main.py`.

Depending on how many sample points we take for each ray, volume rendering can consume a lot of memory on the GPU (especially during the backward pass of gradient descent). Because of this, it usually makes sense to sample a subset of rays from a full image for each training iteration. In order to do this, implement the `get_random_pixels_from_image` method in `ray_utils.py`, invoked here:

```python
xy_grid = get_random_pixels_from_image(cfg.training.batch_size, image_size, camera) # TODO: implement in ray_utils.py
```

##  2.2. Loss and training (5 points)
Replace the loss in `train`

```python
loss = None
```

with mean squared error between the predicted colors and ground truth colors `rgb_gt`.

Once you've done this, you can run train a model with

```bash
python main.py --config-name=train_box
```

This will optimize the position and side lengths of a box, given a few ground truth images with known camera poses (in the `data` folder). Report the center of the box, and the side lengths of the box after training, rounded to the nearest `1/100` decimal place.

##  2.3. Visualization

The code renders a spiral sequence of the optimized volume in `images/part_2.gif`. Compare this gif to the one below, and attach it in your write-up:

![Spiral Rendering of Part 2](ta_images/part_2.gif)


##  3. Optimizing a Neural Radiance Field (NeRF) (30 points)
In this part, you will implement an implicit volume as a Multi-Layer Perceptron (MLP) in the `NeuraRadianceField` class in `implicit.py`. This MLP should map 3D position to volume density and color. Specifically:

1. Your MLP should take in a `RayBundle` object in its forward method, and produce color and density for each sample point in the RayBundle.
2. You should also fill out the loss in `train_nerf` in the `main.py` file.

You will then use this implicit volume to optimize a scene from a set of RGB images. We have implemented data loading, training, checkpointing for you, but this part will still require you to do a bit more legwork than for Parts 1 and 2. You will have to write the code for the MLP yourself --- feel free to reference the NeRF paper, though you should not directly copy code from an external repository.

## Implementation

Here are a few things to note:

1. For now, your NeRF MLP does not need to handle *view dependence*, and can solely depend on 3D position.
2. You should use the `ReLU` activation to map the first network output to density (to ensure that density is non-negative)
3. You should use the `Sigmoid` activation to map the remaining raw network outputs to color
4. You can use *Positional Encoding* of the input to the network to achieve higher quality. We provide an implementation of positional encoding in the `HarmonicEmbedding` class in `implicit.py`.

## Visualization
You can train a NeRF on the lego bulldozer dataset with

```bash
python main.py --config-name=nerf_lego
```

This will create a NeRF with the `NeuralRadianceField` class in `implicit.py`, and use it as the `implicit_fn` in `VolumeRenderer`. It will also train a NeRF for 250 epochs on 128x128 images.

Feel free to modify the experimental settings in `configs/nerf_lego.yaml` --- though the current settings should allow you to train a NeRF on low-resolution inputs in a reasonable amount of time. After training, a spiral rendering will be written to `images/part_3.gif`. Report your results. It should look something like this:

![Spiral Rendering of Part 3](ta_images/part_3.gif)

##  4. NeRF Extras (***Choose at least one!*** More than one is extra credit)

###  4.1 View Dependence (10 pts)

Add view dependence to your NeRF model! Specifically, make it so that emission can vary with viewing direction. You can NeRF or other papers for how to do this effectively --- if you're not careful, your network may overfit to the training images. Discuss the trade-offs between increased view dependence and generalization quality.

###  4.2 Hierarchical Sampling (10 pts)

NeRF employs two networks: a coarse network and a fine network. During the coarse pass, it uses the coarse network to get an estimate of geometry geometry, and during fine pass uses these geometry estimates for better point sampling for the fine network. Implement this hierarchical point-sampling strategy and discuss trade-offs (speed / quality).

###  4.3 High Resolution Imagery (10 pts)

Run NeRF on high-res imagery using the `nerf_lego_highres.yaml` config file. This will take a long time to train -- play around with some hyper parameters (point samples per ray, network capacity) and report your results