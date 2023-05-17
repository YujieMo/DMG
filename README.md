# DMG: Disentangled Multiplex Graph Representation Learning 

This repository contains the reference code for the manuscript ``Disentangled Multiplex Graph Representation Learning" 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)
0. [Testing](#test)

## Installation
* pip install -r requirements.txt 
* Download the datasets
* Download the trained models

## Preparation
Important args:
* `--use_pretrain` Test checkpoints
* `--dataset` acm, imdb, dblp, freebase
* `--custom_key` Node: node classification

## Training
python main.py

## Testing
use_pretrain == 'True'

## Citation
```shell
@InProceedings{Mo_ICML_2023, 
title={Disentangled Multiplex Graph Representation Learning}, 
booktitle={Proceedings of the 40st International Conference on Machine Learning}, 
author={Mo, Yujie and Lei, Yajie and Shen, Jialie and Shi, Xiaoshuang and Shen, Heng Tao and Zhu, Xiaofeng},
year={2023}
}
```
