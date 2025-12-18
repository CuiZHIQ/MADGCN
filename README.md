# MADGCN


## Introduction

This is the official implementation of our paper: [MADGCN: A Meteorology-Aware Spatio-Temporal Graph Convolutional Netowrk for Long-term Air Pollution Forecasting](https://github.com/CuiZHIQ/MADGCN).

In response to escalating global air pollution, air quality forecasting has garnered significant attention. Spatiotemporal graph convolutional networks have emerged as a leading approach. However, existing methods face limitations in modeling long-term dependencies, integrating meteorological variables, and lack large-scale datasets. To address these challenges, we introduce **LargeAQ**, a new large-scale air quality dataset, and propose the **Meteorology-Aware Decoupled Spatio-Temporal Convolutional Network (MADGCN)**. MADGCN jointly addresses long-range temporal modeling and meteorological context integration for accurate and robust air pollution forecasting.

## Model Overview

MADGCN is primarily composed of a dynamic causality discovery module, a causal-aware graph convolution module, and a Patch-Mixer module. Its distinguishing feature is the dynamic causality discovery grounded in the Granger Causality principle, enabling it to capture evolving causal relationships between meteorological conditions and AQI dynamics. These inferred causal structures guide the graph convolution and PatchMixer modules to model spatial interactions and multiscale temporal dependencies together.

## LargeAQ Dataset

We introduce LargeAQ, a publicly available, nationwide air quality dataset spanning eight years (2015–2023) and covering 1,341 monitoring stations across China. This resource is intended to support deeper research into long-term AQI prediction. 

Our dataset is stored in H5 format to avoid the excessive size of CSV files.
> Google Drive Link: https://drive.google.com/file/d/1fBfa4fek4OPZlC-jufs11ocHK3KRXGdX/view?usp=sharing 

We have already preserved the distance-based adjacency matrix in the data structure, which can be easily used. Due to the large scale of the LargeAQ dataset, we recommend users perform temporal downsampling or select a subset of stations based on their specific research needs and computational resources after obtaining the dataset. We recommend referring to the implementations of [LargeST](https://github.com/liuxu77/LargeST) or [BasicTS](https://github.com/zezhishao/BasicTS).

> If you require specific latitude and longitude information for the stations to construct subgraphs for subsets, please send an email to [zhiqing@nuist.edu.cn] to apply for access. Please include the following information in your email: your institution’s name, your full name, and the intended purpose of use. We will ensure a timely response and will share the dataset with you.

| **Dataset**          | **#Stations**                    | **Time span**                         | **Timesteps**                  | **Granularity** | **Coverage**               |
| :------------------- | :------------------------------- | :------------------------------------ | :----------------------------- | :-------------- | :------------------------- |
| BeiJing              | 12                               | 12/05/2014-31/12/2017                 | 5,856                          | 1 h             | City                       |
| ShangHai             | 8                                | 12/05/2014-31/12/2017                 | 5,856                          | 1 h             | City                       |
| ChongQing            | 22                               | 12/05/2014-31/12/2017                 | 5,856                          | 1 h             | City                       |
| KnowAir              | 184                              | 01/01/2014-31/12/2018                 | 11,688                         | 3 h             | Regional                   |
| **LargeAQ (Ours)**   | **1,341**                        | **01/01/2015-31/12/2023**             | **70,128**                     | 1 h             | **National**               |


## Results

### 🏆 Achieves state-of-the-art in Long-Term Air Pollution Forecasting

Extensive experiments against 16 strong baselines demonstrate that MADGCN achieves competitive performance in long-term air pollution forecasting. On the LargeAQ dataset, MADGCN representing substantial improvements over strong baselines. The model shows exceptional stability, with only a 41.2% MAE increase from 12h to 96h prediction horizons.

### 🌟 Effective on High-Pollution Patterns

MADGCN demonstrates strong performance on challenging high-pollution scenarios, such as the 'Heating' and 'Volatile' subsets of the KnowAir dataset, outperforming baselines and validating its effectiveness for predicting high-pollution events.


## 📄 Citation

If you find this project helpful, please cite us:

```bibtex
@article{ma2025causal,
  title={Causal Learning Meet Covariates: Empowering Lightweight and Effective Nationwide Air Quality Forecasting},
  author={Ma, Jiaming and Cui, Zhiqing and Wang, Binwu and Wang, Pengkun and Zhou, Zhengyang and Zhao, Zhe and Wang, Yang}, 
  journal = {International Joint Conference on Artificial Intelligence},
  year={2025}
}
```

## Acknowledgement

We also appreciate the following GitHub repos for their valuable code bases and datasets:

- https://github.com/yuqinie98/PatchTST
- https://github.com/liuxu77/LargeST
- https://github.com/GestaltCogTeam/BasicTS
- https://github.com/PoorOtterBob/CauAir


