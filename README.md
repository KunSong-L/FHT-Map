# FHT-Map

## News
- Jan. 10, 2024: FHT-Map is planned to be open-sourced.

## Introduction
**FHT-Map** is a light-weight framework for building **F**eature-based **H**ybrid **T**opological **M**ap.
Our method is demonstrated to achieve faster relocalization and better path planning capability compared with state-of-the-art topological maps.

**FHT-Map** consists of two types of nodes: main node and support node.
Main nodes store visual information compressed by convolutional neural network and local laser scan data to enhance subsequent relocalization capability.
Support nodes retain a minimal amount of data to ensure storage efficiency while facilitating path planning.

## Quick Start
The detailed code will be released when the paper is accepted.

## Citation
If you use this code for your research, please cite our papers. *https://arxiv.org/abs/2310.13899*

```
@article{song2023fht,
  title={FHT-Map: Feature-based Hierarchical Topological Map for Relocalization and Path Planning},
  author={Song, Kun and Liu, Wenhang and Chen, Gaoming and Xu, Xiang and Xiong, Zhenhua},
  journal={arXiv preprint arXiv:2310.13899},
  year={2023}
}
```
