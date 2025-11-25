# SE360: Semantic Edit in 360° Panoramas via Hierarchical Data Construction

**This repository contains the official implementation of the paper "SE360: Semantic Edit in 360° Panoramas via Hierarchical Data Construction".**

## 📌 Introduction

SE360 is a novel framework for semantic editing in 360° panoramas. By leveraging hierarchical data construction and advanced generative models, SE360 enables high-quality, semantically consistent edits in panoramic images.

**This paper has been accepted by AAAI 2026.**

**Authors:** Haoyi Zhong, Fang-Lue Zhang, Andrew Chalmers, Taehyun Rhee


## 🛠️ Installation

This project has two separate environments:
- **Data Creation**: For dataset preparation and processing
- **Image Editing**: For model training and inference

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/SE360.git
cd SE360
```

### 2. Install dependencies



#### Option A: Data Creation Environment

This environment is used for dataset preparation and processing pipelines (SE360_Base and SE360_HF).

**Step 1: Install Python dependencies**

```bash
conda create -n se360_data python=3.12
conda activate se360_data

# Install PyTorch based on your CUDA version
# For CUDA 12.1 (pytorch 2.5.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.6:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements_datacreation.txt
```

**Step 2: Download checkpoint weights**

Download the required model weights and place them in the `checkpoints/` directory.

```bash
# Download your model weights here
https://drive.google.com/drive/folders/1Hpdf9hEF9HyCnY-BtcjsOypIOq6caZ-L?usp=drive_link

```

**Step 3: Install Grounded-SAM**

```bash
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-12/  # Modify this to your CUDA installation path

python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
```

**Step 4: Install Flash Attention 2**

Download the appropriate wheel file based on your Python, PyTorch, and CUDA versions from the [Flash Attention releases page](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1).

For example, on Linux with PyTorch 2.5, Python 3.12, and CUDA 12.1:

```bash
# Install the wheel
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

> **Note:** Make sure to select the correct wheel file that matches your specific Python version (cp3XX), PyTorch version, and CUDA version.

#### Option B: Image Editing Environment 

This environment is used for training and inference with the SE360 model.

```bash
conda create -n se360 python=3.9
conda activate se360

# Install PyTorch based on your CUDA version
# For CUDA 12.1 (pytorch 2.5.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.6:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126


pip install -r requirements.txt
```


> **Note:** Choose the environment based on your task. For most users interested in running the model, use Option A.


## 📂 Data Preparation

The project expects the dataset to be organized in the `data/` directory. The default configuration looks for Matterport3D data.

### Download Matterport3D

For downloading Matterport3D data, please refer to the download instructions from [PanFusion](https://github.com/chengzhag/PanFusion.git).

### Expected structure (Only show Matterport3D)
```
data/
├── Matterport3D/
│   ├── mp3d_skybox/           # Original skybox images
│   ├── SE360_HF/              
│   └── SE360_Base/            
            
```


## 🚀 Usage

This project uses `LightningCLI` for configuration and execution. You can run training and evaluation using `main.py`.

### Training

To start training the SE360 model:

```bash
python main.py fit \
    --model=SE360 \
    --data=SE360_Base \
```


### Inference / Testing

1. To run inference using a trained checkpoint:

```bash
python main.py test \
    --model=SE360 \
    --data=SE360_Base \
    --ckpt=checkpoints/se360_step_step=001000.ckpt \
    --result_dir results/
```

*   `--ckpt`: Path to the model checkpoint.

2. To evaluate the results:

```bash
python main.py test \
    --model=EvalPanoGen \
    --data=SE360_Base \
    --result_dir logs/your_result_dir/test \

```
*   `--result_dir`: Directory where results will be saved.





