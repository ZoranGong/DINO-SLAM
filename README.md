<p align="center">

  <h2 align="center">[ECCV 2026] DINO-SLAM: DINO-informed RGB-D SLAM for Neural Implicit and Explicit Representations</h2>
  <p align="center">
    <strong>Ziren Gong</strong><sup>1</sup>
    ·
    <strong>Xiaohan Li</strong><sup>2</sup>
    ·
    <strong>Tosi Fabio</strong><sup>1</sup>
    ·
    <strong>Youmin Zhang</strong><sup>1</sup>
    ·
    <strong>Stefano Mattoccia</strong><sup>1</sup>
    ·
    <strong>Matteo Poggi</strong><sup>1</sup>
</p>

   <h3 align="center">

  <h3 align="center"><a href="https://arxiv.org/abs/2507.19474">Paper</a> | <a href="https://zorangong.github.io/DINO-SLAM/">Project Page</a> </h3>
  <div align="center"></div>
</p>



<p align="center">
  <a href="">
    <img src="./assets/frontend_backend.gif" width="100%">
  </a>
</p>

## 📃 Description
<!-- <p align="center">
  <a href="">
    <img src="./assets/architecture.png" width="100%">
  </a>
</p> -->

**DINO-SLAM** is a DINO-informed design strategy to enhance implicit (Neural Radiance Field -- NeRF) and explicit representations (Gaussian Splatting -- GS) in SLAM systems through the more comprehensive semantics understanding enabled by DINO. This latter alone, however, lacks proper 3D geometry understanding, allowing only for marginal improvements. Therefore, we rely on a Scene Geometry Encoder (SGE) to enrich DINO features into geometry-aware DINO features (geoDINO), to better understand those geometric relationships that vanilla DINO features fail to capture. Building upon it, we propose two foundational paradigms for NeRF and GS SLAM systems integrating geoDINO features. Compared to state-of-the-art methods, our DINO-informed pipelines achieve superior performance on the Replica, ScanNet, and TUM datasets.

# 🛠️ Setup
The code has been tested on:

- Ubuntu 20.04 LTS, Python 3.10.15, CUDA 12.2, GeForce RTX 3090/RTX 4090/A6000

## 📦 Repository

Clone the repo with `--recursive` because we have submodules:

```
git clone https://github.com/ZoranGong/DINO-SLAM.git
```

## 💻 Installation
Then setup environment from the provided requirements file,

```
conda create -n dinoslam python=3.10
conda activate dinoslam
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# install pytorch3d offline
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py310_cu116_pyt1121.tar.bz2
conda install pytorch3d-0.7.5-py310_cu116_pyt1121.tar.bz2
```

## 🚀 Usage

Here we will introduce the usage of DINO-SLAM, experiments of the paper and visualization of the results.

### Datasets

  We evaluate our algorithm using the [Replica](https://github.com/facebookresearch/Replica-Dataset) , [TUM-RGBD](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), and [ScanNet](http://www.scan-net.org/) datasets. The download instructions for these four datasets can be obtained from the [LoopSplat](https://github.com/GradientSpaces/loopsplat) project.

  ### Running
  Before running the code, you need to verify that the dataset paths in the `config` directory are correctly specified. To launch the SLAM demo for a specific dataset (e.g., Replica), use the following command:
  ```
  python scripts/dinoslam_mp.py configs/replica/config.py
  ```
  This command will start multi-processed training on the Room0 sequence by default. Note that since both the frontend and backend maintain separate maps in multi-processed mode, this leads to higher GPU memroy usage. 
  
  We have also implemented a single-processed demo with lower memory consumption, which can be launched using the following command:
  ```
  python scripts/dinoslam.py configs/replica/config.py
  ```
  After training completes, the corresponding results will be generated in the output directory by default.

  For faster running, use the following command:
  ```
  python scripts/dinoslam_mp.py configs/replica/config_fast.py
  ```

  ### Evaluation 
  During training, *PSNR*, *SSIM*, *LPIPS*, and *ATE-RMSE* metrics are computed by default.
  

## 🎓 Citation
If you find our work helpful in your research, please consider citing: 
```
@inproceedings{gong2026dino,
  title={DINO-SLAM: dino-informed RGB-D SLAM for neural implicit and explicit representations},
  author={Gong, Ziren and Li, Xiaohan and Tosi, Fabio and Zhang, Youmin and Mattoccia, Stefano and Wu, Jun and Poggi, Matteo},
  booktitle={European Conference on Computer Vision},
  pages={158--177},
  year={2026},
  organization={Springer}
}
```
