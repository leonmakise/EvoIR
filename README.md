<div align="center">
<h1> <img src="/EvoIR-logo.svg" height="35">EvoIR: Towards All-in-One Image Restoration via Evolutionary Frequency Modulation</h1>

</div>

#### [Jiaqi Ma]()\*, [Shengkai Hu]()\*, [Xu Zhang](), [Jun Wan](), [Jiaxing Huang](), [Lefei Zhang]() and [Salman Khan]()
\* Equally contribution

**Mohamed bin Zayed University of Artificial Intelligence**, **Zhongnan University of Economics and Law**, 

**Wuhan University**, **Nanyang Technological University**

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="EvoIR">
</p>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2512.05104)
[![hf_model](https://img.shields.io/badge/🤗-Model-blue.svg)](https://huggingface.co/leonmakise/EvoIR)

---

## Updates
This project is under active development, please stay tuned! ☕

**Dec 4, 2025:** We release the preprint version of EvoIR (https://arxiv.org/abs/2512.05104).

**Aug 15, 2025:** We've released EvoIR! We wish this work would inspire more works on all-in-one image restoration tasks.


## Highlights
![](figures/main_figure.jpg)

* We propose EvoIR; to the best of our knowledge within AiOIR, it is the first framework that leverages an evolutionary algorithm for loss weighting, together with frequency-aware modulation. EvoIR attains state-of-the-art performance across multiple benchmarks and remains robust to diverse degradation.
* We introduce a Frequency-Modulated Module (FMM) that explicitly separates features into high- and low-frequency components and dynamically modulates each branch to target finegrained textures and structural smoothness under complex degradation.
* We present an Evolutionary Optimization Strategy (EOS), a population-based mechanism with modest overhead that automatically identifies and adapts optimal loss-weight configurations for AiOIR, improving convergence and balancing perceptual quality without manual tuning.


## Getting Started

### Requirement
Plz follow the env.yaml environment.



### Data Preparation
#### Download Path
- Download the denoising dataset from [BSD](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/).
- Download the low-light enhancement dataset from [LoL](https://daooshee.github.io/BMVC2018website/). 
- Download the deraining dataset from [Rain100L](https://github.com/nnUyi/DerainZoo/blob/master/DerainDatasets.md). 
- Download the deblurring dataset from [GoPro](https://seungjunnah.github.io/Datasets/gopro). 
- Download the dehazing dataset from [OTS](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)

#### Dataset Structure
We recommend the dataset directory structure to be the following:

```bash
$EvoIR/datasets/
    train/
        Denoise/
        Low-light/
            gt/
            input/
        Derain/
            gt/
            rainy/
        Deblur/
            target/
            input/
        Dehaze/
            original/
            synthetic/
    test/
        Denoise/
            bsd68/
        Low-light/
            gt/
            input/
        Derain/
            rain100L/
                target/
                input/
        Deblur/
            target/
            input/
        Dehaze/
            target/
            input/

```

### Training
Run the following command:
```shell
python train.py
```


### Evaluation
Run the following command:
```shell
python test.py
```


## Experimental Results
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolop-you-only-look-once-for-panoptic-driving/traffic-object-detection-on-bdd100k)](https://paperswithcode.com/sota/traffic-object-detection-on-bdd100k?p=yolop-you-only-look-once-for-panoptic-driving) -->

#### Performance on 3-Tasks Image Restorations Tasks

| Method                | Denoising (CBSD68) σ=15 |       σ=25      |       σ=50      | Dehazing (SOTS) | Deraining (Rain100L) |     Average     | Params (M) |
| :-------------------- | :---------------------: | :-------------: | :-------------: | :-------------: | :------------------: | :-------------: | ---------: |
| AirNet (CVPR'22)      |       33.92/0.932       |   31.26/0.888   |   28.00/0.797   |   27.94/0.962   |      34.90/0.967     |   31.20/0.910   |       8.93 |
| IDR (CVPR'23)         |       33.89/0.931       |   31.32/0.884   |   28.04/0.798   |   29.87/0.970   |      36.03/0.971     |   31.83/0.911   |      15.34 |
| ProRes (arXiv'23)     |       32.10/0.907       |   30.18/0.863   |   27.58/0.779   |   28.38/0.938   |      33.68/0.954     |   30.38/0.888   |     370.63 |
| PromptIR (NeurIPS'23) |       33.98/0.933       |   31.31/0.888   |   28.06/0.799   |   30.58/0.974   |      36.37/0.972     |   32.06/0.913   |      32.96 |
| NDR (TIP'24)          |       34.01/0.932       |   31.36/0.887   |   28.10/0.798   |   28.64/0.962   |      35.42/0.969     |   31.51/0.910   |      28.40 |
| Gridformer (IJCV'24)  |       33.93/0.931       |   31.37/0.887   |   28.11/0.801   |   30.37/0.970   |      37.15/0.972     |   32.19/0.912   |      34.07 |
| InstructIR (ECCV'24)  |       34.15/0.933       |   31.52/0.890   |   28.30/0.804   |   30.22/0.959   |      37.98/0.978     |   32.43/0.913   |      15.84 |
| Up-Restorer (AAAI'25) |       33.99/0.933       |   31.33/0.888   |   28.07/0.799   |   30.68/0.977   |      36.74/0.978     |   32.16/0.915   |      28.01 |
| Perceive-IR (TIP'25)  |       34.13/0.934       |   31.53/0.890   |   28.31/0.804   |   30.87/0.975   |      38.29/0.980     |   32.63/0.917   |      42.02 |
| AdaIR (ICLR'25)       |       34.12/0.935       |   31.45/0.892   |   28.19/0.802   |   31.06/0.980   |      38.64/0.983     |   32.69/0.918   |      28.77 |
| MoCE-IR (CVPR'25)     |       34.11/0.932       |   31.45/0.888   |   28.18/0.800   |   31.34/0.979   |      38.57/0.984     |   32.73/0.917   |      25.35 |
| DFPIR (CVPR'25)       |       34.14/0.935       |   31.47/0.893   |   28.25/0.806   |   31.87/0.980   |      38.65/0.982     |   32.88/0.919   |      31.10 |
| **EvoIR** (Ours)      |     **34.14/0.937**     | **31.48/0.896** | **28.23/0.811** | **32.08/0.982** |    **39.07/0.985**   | **33.00/0.922** |  **36.68** |



**Notes**: 
- The works we used for reference including `Uformer`([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf),[code](https://github.com/ZhendongWang6/Uformer)), `MPRNet`([paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf),[code](https://github.com/swz30/MPRNet)), `MIRNet-v2`([paper](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/),[code](https://github.com/swz30/MIRNetv2)), `Restormer`([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf),[code](https://github.com/swz30/Restormer)), `MAXIM`([paper](https://openaccess.thecvf.com//content/CVPR2022/papers/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.pdf),[code](https://github.com/google-research/maxim)) and `Painter`([paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Images_Speak_in_Images_A_Generalist_Painter_for_In-Context_Visual_CVPR_2023_paper.pdf),[code](https://github.com/baaivision/Painter)).

- More experimental results are listed in the paper!
---






## Acknowledgement
This project is based on [PromptIR](https://github.com/va1shn9v/PromptIR), [AdaIR](https://github.com/c-yn/AdaIR), [Perceive-IR](https://github.com/House-yuyu/Perceive-IR) and [ProRes](https://github.com/leonmakise/ProRes). Thanks for their wonderful work!
