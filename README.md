# SDD-RT-DETR-scallop-doneness-detection

🎯 Model Overview

SDD‑RT‑DETR is a lightweight, real‑time object detection model designed specifically for scallop doneness classification in an industrial shucking line. SDD-RT-DETR integrates an enhanced backbone centered on the self-developed module HierarchicalRepBlock, a frequency-domain self-attetion module AIFI-EDFFN, a neck featuring the self-developed EfficientBalanceFusion module as the feature fusion unit and the Converse2DC3 module as the feature extraction unit, and a loss function named Wise-DIoU.

Trained on a scallop doneness dataset, SDD‑RT‑DETR reliably detects and classifies shucked scallops in challenging production environments such as stacking, occlusion, and lighting variations into three categories:

Raw (under‑cooked)
Medium (properly cooked)
Cooked (over‑cooked)

This model serves as the visual perception core of a feedback temperature control system, achieving 95.5% accuracy, 93.6% recall, and 96.1% mAP50 on our scallop doneness dataset. It provides real‑time doneness information that enables automatic adjustment of the water‑bath temperature, ensuring consistent product quality while reducing manual intervention. Designed for deployment on edge‑class hardware, it achieves < 50 ms inference per image with a lightweight footprint, making it suitable for high‑speed production environments.



📁 Project Structure


├── improved_modules/     # All custom improvement modules

├── models/               # Model definitions

├── nn/                   # Neural network components

├── engine/               # Training and inference engine

├── cfg/                  # Configuration files

├── utils/                # Utility functions

├── hub/                  # Pretrained model hub

├── data/                 # Data loading utilities

├── data.yaml             # Dataset configuration

├── cfg/SDD-RT-DETR.yaml      # Model architecture definition

├── train.py              # Training script

├── detect.py             # Detection/inference script

└── __init__.py           # Package initialization



🚀 Quick Start

Before running training or inference, please modify the dataset path in `data.yaml` to point to your own dataset location.

Model configuration: architecture details are defined in SDD-RT-DETR.yaml.

Custom modules: all improved components (HierarchicalRepBlock, AIFI-EDFFN, EfficientBalanceFusionModule, Converse2DC3, Wise-DIoU) are placed in the improved_modules/ folder.

Training: run train.py to start training on your dataset.

Inference: run detect.py for detection on images or video streams.



📦 Environment & DependenciesEnvironment & Dependencies

This project is developed and tested under the following environment:

Python: 3.10.14

PyTorch: 2.2.2+cu121

Torchvision: 0.17.2+cu121

timm: 1.0.7

mmcv: 2.2.0

mmengine: 0.10.4

triton: 3.2.0



📦 Installation

The required packages are consistent with the official Ultralytics RT-DETR setup. Please refer to (https://docs.ultralytics.com/models/rtdetr/) for more details

Additionally, you may need to install the following extra dependencies:

pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python prettytable
pip install torch-dct==0.1.6



📝 Citation

If you find this work useful, please consider citing our repository.



📁 Sample Images: A subset of 50 images from the scallop doneness dataset is available at this link.
[https://pan.baidu.com/s/1xOBTdDK4UGetyuL5eng0Ig?pwd=7j8s]



🖼️ Detection Results

<img width="312" height="236" alt="image" src="https://github.com/user-attachments/assets/6d3bb78d-f2a6-4dc4-816d-bab8ad8df8bf" />

<img width="312" height="236" alt="image" src="https://github.com/user-attachments/assets/4d676a5d-7bc2-42f2-b1f6-4713f048dff3" />

<img width="312" height="236" alt="image" src="https://github.com/user-attachments/assets/3050dd11-698f-451c-b7ae-61cecd1dcff7" />










