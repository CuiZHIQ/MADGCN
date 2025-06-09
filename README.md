# MADGCN


## Introduction

This is the official implementation of our paper: [MADGCN: A Meteorology-Aware Spatio-Temporal Graph Convolutional Netowrk for Long-term Air Pollution Forecasting](https://github.com/CuiZHIQ/MADGCN).

In response to escalating global air pollution, air quality forecasting has garnered significant attention. Spatiotemporal graph convolutional networks have emerged as a leading approach. However, existing methods face limitations in modeling long-term dependencies, integrating meteorological variables, and lack large-scale datasets. To address these challenges, we introduce **LargeAQ**, a new large-scale air quality dataset, and propose the **Meteorology-Aware Decoupled Spatio-Temporal Convolutional Network (MADGCN)**. MADGCN jointly addresses long-range temporal modeling and meteorological context integration for accurate and robust air pollution forecasting.

## Model Overview

MADGCN is primarily composed of a dynamic causality discovery module, a causal-aware graph convolution module, and a Patch-Mixer module. Its distinguishing feature is the dynamic causality discovery grounded in the Granger Causality principle, enabling it to capture evolving causal relationships between meteorological conditions and AQI dynamics. These inferred causal structures guide the graph convolution and PatchMixer modules to model spatial interactions and multiscale temporal dependencies together.

## LargeAQ Dataset

We introduce LargeAQ, a publicly available, nationwide air quality dataset spanning eight years (2015–2023) and covering 1,341 monitoring stations across China. This resource is intended to support deeper research into long-term AQI prediction.

| **Dataset**          | **#Stations**                    | **Time span**                         | **Timesteps**                  | **Granularity** | **Coverage**               |
| :------------------- | :------------------------------- | :------------------------------------ | :----------------------------- | :-------------- | :------------------------- |
| BeiJing              | 12                               | 12/05/2014-31/12/2017                 | 5,856                          | 1 h             | City                       |
| ShangHai             | 8                                | 12/05/2014-31/12/2017                 | 5,856                          | 1 h             | City                       |
| ChongQing            | 22                               | 12/05/2014-31/12/2017                 | 5,856                          | 1 h             | City                       |
| KnowAir              | 184                              | 01/01/2014-31/12/2018                 | 11,688                         | 3 h             | Regional                   |
| **LargeAQ (Ours)**   | **1,341**                        | **01/01/2015-31/12/2023**             | **70,128**                     | 1 h             | **National**               |

## Getting Started

1.  Install requirements.

    ```
    pip install -r requirements.txt
    ```

2.  Download data. You can download our LargeAQ dataset from our GitHub repository. Create a separate folder `./dataset` and put all the files in the directory. The LargeAQ dataset and the code for constructing the dual graph will be open sourced after the paper is accepted.

3.  Training. All the scripts are in the directory `./scripts/MADGCN`. For example, if you want to get the forecasting results for the LargeAQ dataset, just run the following command. You can check `result.txt` to see the results once training is done, and the log file is in `./logs/largeaq/*.log`.

    ```
    sh ./scripts/MADGCN/largeaq.sh
    ```

## Results

### 🏆 Achieves state-of-the-art in Long-Term Air Pollution Forecasting

Extensive experiments against 16 strong baselines demonstrate that MADGCN achieves competitive performance in long-term air pollution forecasting. On the LargeAQ dataset, MADGCN attains MAE/RMSE of 18.57/27.79 for 24-hour prediction, representing substantial improvements over strong baselines. The model shows exceptional stability, with only a 41.2% MAE increase from 12h to 96h prediction horizons.

### 🌟 Effective on High-Pollution Patterns

MADGCN demonstrates strong performance on challenging high-pollution scenarios, such as the 'Heating' and 'Volatile' subsets of the KnowAir dataset, outperforming baselines and validating its effectiveness for predicting high-pollution events.

## Acknowledgement

We also appreciate the following GitHub repos for their valuable code bases and datasets:

- https://github.com/yuqinie98/PatchTST
- https://github.com/GestaltCogTeam/BasicTS


