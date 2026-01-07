# Few-shot Cross-episode Adaptive Memory for Metal Surface Defect Semantic Segmentation


## 📋 Project Overview
This project introduces an **Episode-adaptive Memory Network (EAMNet)** method for subtle variances between episodes
during training.

## ✨ Key Features
- An EAMU generates a cross-episode adaptive factor to exploit the semantic dependence across episodes for metal surface defect regions.
- We propose a CAM and a GRMAP leveraging hierarchical features and global response normalization respectively to accomplish fine-grained segmentation.
- To accelerate cross-episode semantic learning and enhance generalization, we introduce an AD that transfers fine-grained semantic attention from high-resolution features.
- Our EMANet sets new state-of-the-art results on both standard benchmarks: **Surface Defect-4^i** and **FSSD-12**.

## 📂 Datasets
This project utilizes the following two public metal surface defect datasets for training and evaluation:

| Dataset | Description                                                                                                                                                                                          | Source | Notes |
|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-------|
| **Surface Defect-4^i** | There are a total of 12 different classes of surface defects in the dataset. Each class includes defective images, groundtruth (GT), and a large number of normal images.                            | [Surface Defect-4^i GitHub](https://github.com/bbbbby-99/TGRNet-Surface-Defect-Segmentation) | 200×200 pixel images with pixel-level annotations. |
| **FSSD-12** | there are twelve S3D classes in FSSD-12, including abrasion-mask, iron-sheet ash, liquid, oxide-scale, oil-spot, water-spot, patch, punching, red-iron sheet, roll-printing, scratch, and inclusion. | [FSSD-12 GitHub](GitHub) | 200×200 pixel images with pixel-level annotations. |

> Note: Please comply with each dataset's license agreement and ensure proper placement in the `data/` directory.

## 🛠 Environment Setup

### Prerequisites
- Python 3.12+
- PyTorch 2.6+
- CUDA 11.0+ (for GPU training)