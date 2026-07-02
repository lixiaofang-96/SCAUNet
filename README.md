# SCAUNet

This is the official code for "SCAUNet: Step-Size Consistent ADMM Unfolding Network for Low-Light Image Enhancement."

## Abstract

Low-light image enhancement aims to restore visually pleasing normal-light images from degraded low-light observations. Most existing methods handle luminance variation from the enhancement perspective. As a result, the degradation process from a normal-light image to a low-light observation is usually not explicitly characterized. In addition, degradation-oriented optimization is often computationally expensive due to repeated iterative updates. To address these issues, based on the alternating direction method of multipliers (ADMM), a degradation-oriented step-size consistent unfolding network SCAUNet is proposed. Specifically, a low-light image is modeled as the element-wise product of a normal-light image and a luminance degradation operator, together with additive noise. Based on this formulation, low-light enhancement is converted into the joint estimation of the target image and the degradation operator. Then, a state-based one step ADMM solver is developed, and a step-size consistency constraint is introduced to improve the reliability of one step unfolding. Extensive experiments on LOL-v1 and LOL-v2 demonstrate the effectiveness of the proposed SCAUNet. Compared with existing state-of-the-art methods, SCAUNet yields better enhancement quality, especially in preserving image structures, correcting illumination, and suppressing artifacts. Strong generalization ability is also verified on four no-reference low-light datasets, and promising results are obtained on single image exposure correction.

### 1. Download the project.

Please run the following command to ensure that you deploy our project locally.

```python
git clone https://github.com/lixiaofang-96/SCAUNet.git
```

### 2. Create environment.

Please note that since the "causal_conv1d" package is only available on Linux systems, ensure that your operating environment is Linux.

### 2.1 Create Conda environment.

To prevent any discrepancies between your environment and ours, we recommend that you choose the same virtual environment as us. You can directly install the environment we have packaged for you, or choose to follow our tutorial to install it step by step.

```python
conda create -n SCAUNet python=3.11
conda activate SCAUNet
```

### 2.2 Install dependencies.

```python
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu124
pip install packaging
pip install timm==1.0.11
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==3.0.0
pip install causal_conv1d==1.5.0.post8
pip install mamba_ssm==2.2.4
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy
pip install opencv-python joblib natsort tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
pip install fvcore
```

#### 2.3 Install BasicSR.

```python
cd /SCAUNet/
python setup.py develop --no_cuda_ext
```

### 3. Prepare the dataset.

Change the dataset paths in the yml file to your own local paths.


### 4. Test

The pretrained weights will be released later.

```
# activate the environment
conda activate SCAUNet

# LOL-v1
python3 Enhancement/test_ADMMNet.py --opt Options/LOL_v1.yml --weights 权重路径 --dataset ADMMNet_LOL_v1 --GT_mean

# LOL-v2-real
python3 Enhancement/test_ADMMNet.py --opt Options/LOL_v2_real.yml --weights 权重路径 --dataset ADMMNet_LOL_v2_real
```


### 5. Model parameters and FLOPS evaluation.

If you want to see the parameter count of the model,  simply run `ADMMNet_arch`.

### 6. Train

Please ensure that you have fully completed the environment setup and can correctly infer the parameters and floating points.

```
# LOL-v1
python3 basicsr1/train.py --opt Options/LOL_v1.yml

# LOL-v2-real
python3 basicsr1/train.py --opt Options/LOL_v2_real.yml
```

