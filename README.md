# EfficientNet
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result
<img width="2480" height="1914" alt="efficientnet_training_curve" src="https://github.com/user-attachments/assets/228998f9-5a9c-4a30-8617-6ea6509af1f5" />

<img width="560" height="559" alt="image" src="https://github.com/user-attachments/assets/1fa0b499-7aaf-42cb-b423-8a4db9e264fc" />

参数/FLOPS与精度曲线：
<img width="1015" height="399" alt="image" src="https://github.com/user-attachments/assets/ee7f9989-7bd9-4489-a2e4-2ad349365e89" />



---

## 简介
EfficientNet 是由谷歌团队于 2019 年提出的**轻量化高精度卷积神经网络**，相关成果发表于《EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks》。针对传统卷积网络与早期轻量化模型存在的痛点：单独放大网络深度、宽度或输入分辨率易造成算力浪费、精度增益边际递减，MobileNet 系列仅靠结构轻量化难以进一步提升精度，EfficientNet 首次提出**复合缩放策略**，以统一系数同步缩放网络深度、宽度与输入分辨率，在极低参数量与计算量下实现远超同规模模型的识别精度。其核心创新包含三大模块：**MBConv倒置瓶颈模块**、**SE通道注意力机制**、**DropConnect随机深度正则化**，同时采用 Swish(SiLU) 激活函数、通道8倍数约束等工程优化设计。该模型打破了传统模型手工扩容的固有思路，以B0为基准衍生出B0~B7一系列梯度模型，可灵活适配移动端、边缘设备、服务器端不同算力场景，广泛应用于图像分类、目标检测、语义分割等计算机视觉任务，成为后续高精度轻量化网络设计的标杆骨架，极大推动了轻量CNN在工业落地与学术研究中的发展。


## 架构
EfficientNet 整体为**复合缩放驱动+多阶段MBConv堆叠的端到端高精度轻量化卷积神经网络**，整体分为「Stem初始卷积模块」「多阶段MBConv特征提取模块」和「全局池化+分类Head输出模块」三大核心部分，原论文标准以EfficientNet-B0为基准，输入为224×224分辨率的3通道RGB图像，通过复合缩放衍生B1~B7多尺度模型，适配不同精度与算力需求，具体结构与核心设计如下：
- **Stem初始模块**：网络首层采用标准3×3普通卷积、步长为2，完成原始图像浅层特征提取与首次下采样，搭配BN批量归一化与Swish激活函数，替代传统ReLU获得更强非线性表达，为后续MBConv模块提供高质量基础特征。
- **轻量化特征提取模块（核心）**：网络主体划分7个特征Stage，全程堆叠**MBConv倒置瓶颈卷积块**，模块内部遵循「1×1逐点卷积升维 → 3×3/5×5深度卷积空间特征提取 → SE通道注意力加权 → 1×1逐点卷积降维」的固定范式；引入**expand_ratio通道扩张倍数**控制模块内部特征升维幅度，**num_layers模块堆叠层数**控制每个Stage的网络深度；同时嵌入DropConnect随机深度，随机关闭残差捷径连接，避免深层网络过拟合。通过控制宽度系数、深度系数实现所有Stage通道数、堆叠层数的等比例复合缩放。
- **分类输出模块**：后端采用1×1卷积统一映射至高维特征空间，配合全局平均池化压缩特征图尺寸，摒弃冗余全连接层减少参数量；末端通过Dropout正则化+单层全连接层映射分类维度，原论文ImageNet任务输出1000维类别得分，结构简洁且泛化能力极强。

该架构重构了卷积网络模型缩放范式，以复合缩放为核心、MBConv+SE注意力为基础、DropConnect为正则化手段，兼顾**轻量化、高精度、低算力**三大优势，既可作为独立分类模型使用，也可作为骨干网络迁移至目标检测、分割等下游视觉任务，是现代轻量化高精度网络的经典代表。

联合缩放架构(Compound Scaling)：
<img width="1055" height="445" alt="image" src="https://github.com/user-attachments/assets/8df75628-c161-46ac-aa21-8346d9f53399" />

联合缩放重要性：
<img width="1029" height="330" alt="image" src="https://github.com/user-attachments/assets/f6b57b13-d416-4579-9fa2-a330e4b82395" />

网络架构：
<img width="1041" height="525" alt="image" src="https://github.com/user-attachments/assets/1e0582e0-f85d-4677-bf3d-88df6837e64a" />



**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要调整Stem初始卷积下采样步长、控制特征图尺寸不被过度压缩），但核心架构**复合缩放策略、MBConv倒置瓶颈、SE通道注意力、DropConnect正则化、Swish激活**完全保留，严格复现原版EfficientNet核心设计思想。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

---

## Introduction
EfficientNet is a high-precision lightweight convolutional neural network proposed by the Google team in 2019, published in the paper *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. Aiming at the drawbacks of traditional CNNs and early lightweight models: blindly increasing network depth, width or input resolution leads to computational resource waste and diminishing accuracy gains; MobileNet series only relies on structural lightweighting and is difficult to further improve performance. EfficientNet firstly proposes a **compound scaling strategy**, which synchronously scales network depth, width and input resolution with a unified coefficient. It achieves state-of-the-art accuracy with extremely low parameters and computational cost. Its core innovations include three key components: **MBConv inverted bottleneck module**, **SE channel attention mechanism**, and **DropConnect stochastic depth regularization**. It also adopts Swish(SiLU) activation function and channel constraint of multiple of 8 for engineering optimization. Derived from the baseline B0, a series of gradient models from B0 to B7 are formed, which can flexibly adapt to mobile terminals, edge devices and server ends with different computing power. It is widely used in image classification, object detection, semantic segmentation and other visual tasks, and has become a benchmark backbone for the design of subsequent high-precision lightweight networks.


## Architecture
The overall structure of EfficientNet is an end-to-end high-precision lightweight convolutional neural network driven by compound scaling and stacked with multi-stage MBConv. It is divided into three core parts: Stem initial convolution module, multi-stage MBConv feature extraction module, and global pooling & classification Head output module. EfficientNet-B0 is used as the baseline in the original paper with 224×224 RGB input images, and B1~B7 multi-scale models are derived through compound scaling to adapt to different accuracy and computing power requirements.
- **Initial Stem Module**: The first layer adopts standard 3×3 ordinary convolution with stride=2, completing shallow feature extraction and initial downsampling. Combined with BN normalization and Swish activation function, it obtains stronger nonlinear expression than ReLU and provides high-quality basic features for subsequent MBConv modules.
- **Lightweight Feature Extraction Module (Core)**: The main body of the network is divided into 7 feature stages, stacked with **MBConv inverted bottleneck blocks**. The internal paradigm is fixed: 1×1 pointwise convolution channel expansion → 3×3/5×5 depthwise convolution spatial feature extraction → SE channel attention feature weighting → 1×1 pointwise convolution channel compression. The **expand_ratio** controls the channel expansion ratio inside the module, and **num_layers** controls the stacking number of modules in each stage. DropConnect stochastic depth is embedded to randomly close residual shortcut connections to prevent overfitting of deep networks. The width and depth coefficients realize proportional compound scaling of channel number and stacking layers for all stages.
- **Classification Output Module**: A 1×1 convolution is used to uniformly map high-dimensional feature space, and global average pooling compresses feature map size. Redundant fully connected layers are abandoned to reduce parameters. The final Dropout regularization plus single fully connected layer maps to classification dimensions, outputting 1000-dimensional category scores for ImageNet tasks with concise structure and strong generalization ability.

Compound Scaling Architecture:
<img width="1055" height="445" alt="image" src="https://github.com/user-attachments/assets/8df75628-c161-46ac-aa21-8346d9f53399" />

Importance of Compound Scaling:
<img width="1029" height="330" alt="image" src="https://github.com/user-attachments/assets/f6b57b13-d416-4579-9fa2-a330e4b82395" />

Network Architecture:
<img width="1041" height="525" alt="image" src="https://github.com/user-attachments/assets/1e0582e0-f85d-4677-bf3d-88df6837e64a" />

**Note:** We use the CIFAR-10 dataset with 10 classification categories. Since the 32×32 image size of CIFAR-10 is much smaller than the 224×224 input in the original paper, slight adjustments are made to the downsampling stride of the initial Stem convolution to avoid excessive feature compression. However, the core designs of **compound scaling strategy, MBConv inverted bottleneck, SE channel attention, DropConnect regularization and Swish activation** are completely consistent with the original EfficientNet.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

---
## 原文章 | Original article
Tan M, Le Q V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks[C]//International Conference on Machine Learning. PMLR, 2019: 6105-6114.
