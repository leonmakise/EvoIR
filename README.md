<div align="center">
<h1> EvoIR 🌆</h1>
<h3> EvoIR: Towards All-in-One Image Restoration via Evolutionary Frequency Modulation </h3>

</div>


## Updates
This project is under active development, please stay tuned! ☕

**Aug 15, 2025:** We've released EvoIR! We wish this work would inspire more works on all-in-one image restoration tasks.


## Highlights
![](figures/main_figure.jpg)

* We propose EvoIR, a novel All-in-One Image Restoration framework that integrates frequency-aware modulation and evolutionary loss optimization. EvoAdaIR establishes SOTA performance across multiple restoration benchmarks, demonstrating strong robustness to diverse scenarios.

* We design Adaptive Frequency-Modulated Module (AFMM), explicitly designed to separate features into distinct high- and low-frequency components. AFMM dynamically modulates each frequency branch to achieve targeted enhancement of fine-grained textures and structural smoothness under complex degradation conditions.

* We introduce Evolutionary Optimization Strategy (EOS) , a population-based search mechanism that automatically identifies and dynamically adapts the optimal loss weight configurations. EOS facilitates better convergence and improved balance between different degradation types and perceptual quality without manual tuning.

## Getting Started

### Requirement
Plz follow the env.yaml environment.



### Data Preparation
#### Download Path
- Download the denoising dataset from [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/).
- Download the low-light enhancement dataset from [LoL](https://daooshee.github.io/BMVC2018website/). 
- Download the deraining dataset from [Synthetic Rain Datasets](https://github.com/swz30/MPRNet/blob/main/Deraining/Datasets/README.md). 
- Download the deblurring dataset from [Synthetic Blur Datasets](https://github.com/swz30/MPRNet/blob/main/Deblurring/Datasets/README.md). 


#### Dataset Structure
We recommend the dataset directory structure to be the following:

```bash
$ProRes/datasets/
    denoise/
        train/
        val/
    enhance/
        our485/
            low/
            high/
        eval15/
            low/
            high/
    derain/
        train/
            input/
            target/
        test/
            Rain100H/
            Rain100L/
            Test100/
            Test1200/
            Test2800/
    deblur/
        train/
            input/
            target/
        test/
            GoPro/
            HIDE/
            RealBlur_J/
            RealBlur_R/

    target-derain_train.json
    gt-enhance_lol_train.json
    groundtruth-denoise_ssid_train448.json
    groundtruth_crop-deblur_gopro_train.json
    
    target-derain_test_rain100h.json
    gt-enhance_lol_eval.json
    groundtruth-denoise_ssid_val256.json
    groundtruth-deblur_gopro_val.json
```

### Training

```shell
python train.py
```



### Evaluation

Run the following command:
```shell
python test.py
```

<!--
## Experimental Results
<!-- [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/yolop-you-only-look-once-for-panoptic-driving/traffic-object-detection-on-bdd100k)](https://paperswithcode.com/sota/traffic-object-detection-on-bdd100k?p=yolop-you-only-look-once-for-panoptic-driving) -->

#### Performance on Image Restorations Tasks
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="3"></th>
    <th class="tg-c3ow" colspan="2" align="center">denoising</th>
    <th class="tg-c3ow" colspan="2" align="center">deraining</th>
    <th class="tg-c3ow" colspan="2" align="center">enhance</th>
    <th class="tg-c3ow" colspan="2" align="center">deblurring</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="2" align="center">SIDD</th>
    <th class="tg-c3ow" colspan="2" align="center">5 datasets</th>
    <th class="tg-c3ow" colspan="2" align="center">LoL</th>
    <th class="tg-c3ow" colspan="2" align="center">4 datasets</th>
  </tr>
  <tr>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" colspan="9" align="center">Task-specific models</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Uformer</td>
    <td class="tg-c3ow">39.89</td>
    <td class="tg-c3ow">0.960</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">32.31</td>
    <td class="tg-c3ow">0.941</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MPRNet</td>
    <td class="tg-c3ow">39.71</td>
    <td class="tg-c3ow">0.958</td>
    <td class="tg-c3ow">32.73</td>
    <td class="tg-c3ow">0.921</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">33.67</td>
    <td class="tg-c3ow">0.948</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MIRNet-v2</td>
    <td class="tg-c3ow">39.84</td>
    <td class="tg-c3ow">0.959</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">24.74</td>
    <td class="tg-c3ow">0.851</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Restormer</td>
    <td class="tg-c3ow">40.02</td>
    <td class="tg-c3ow">0.960</td>
    <td class="tg-c3ow">33.96</td>
    <td class="tg-c3ow">0.935</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">32.32</td>
    <td class="tg-c3ow">0.935</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MAXIM</td>
    <td class="tg-c3ow">39.96</td>
    <td class="tg-c3ow">0.960</td>
    <td class="tg-c3ow">33.24</td>
    <td class="tg-c3ow">0.933</td>
    <td class="tg-c3ow">23.43</td>
    <td class="tg-c3ow">0.863</td>
    <td class="tg-c3ow">34.50</td>
    <td class="tg-c3ow">0.954</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="9" align="center">Universal models</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Painter</td>
    <td class="tg-c3ow">38.88</td>
    <td class="tg-c3ow">0.954</td>
    <td class="tg-c3ow">29.49</td>
    <td class="tg-c3ow">0.868</td>
    <td class="tg-c3ow">22.40</td>
    <td class="tg-c3ow">0.872</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ViT-Large</td>
    <td class="tg-c3ow">39.28</td>
    <td class="tg-c3ow">0.967</td>
    <td class="tg-c3ow">30.75</td>
    <td class="tg-c3ow">0.893</td>
    <td class="tg-c3ow">21.69</td>
    <td class="tg-c3ow">0.850</td>
    <td class="tg-c3ow">20.57</td>
    <td class="tg-c3ow">0.680</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ProRes</td>
    <td class="tg-c3ow">39.28</td>
    <td class="tg-c3ow">0.967</td>
    <td class="tg-c3ow">30.67</td>
    <td class="tg-c3ow">0.891</td>
    <td class="tg-c3ow">22.73</td>
    <td class="tg-c3ow">0.877</td>
    <td class="tg-c3ow">28.03</td>
    <td class="tg-c3ow">0.897</td>
  </tr>
</tbody>
</table>
**
-->

**Notes**: 
- The works we used for reference including `Uformer`([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.pdf),[code](https://github.com/ZhendongWang6/Uformer)), `MPRNet`([paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.pdf),[code](https://github.com/swz30/MPRNet)), `MIRNet-v2`([paper](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/),[code](https://github.com/swz30/MIRNetv2)), `Restormer`([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf),[code](https://github.com/swz30/Restormer)), `MAXIM`([paper](https://openaccess.thecvf.com//content/CVPR2022/papers/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.pdf),[code](https://github.com/google-research/maxim)) and `Painter`([paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Images_Speak_in_Images_A_Generalist_Painter_for_In-Context_Visual_CVPR_2023_paper.pdf),[code](https://github.com/baaivision/Painter)).

- More experimental results are listed in the paper!
---

<!-- #### ProRes and the vanilla task-specific models:


<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="3"></th>
    <th class="tg-c3ow" colspan="2">denoising</th>
    <th class="tg-c3ow" colspan="2">deraining</th>
    <th class="tg-c3ow" colspan="2">enhance</th>
    <th class="tg-c3ow" colspan="2">deblurring</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="2">SIDD</th>
    <th class="tg-c3ow" colspan="2">5 datasets</th>
    <th class="tg-c3ow" colspan="2">LoL</th>
    <th class="tg-c3ow" colspan="2">4 datasets</th>
  </tr>
  <tr>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
    <th class="tg-c3ow">PSNR</th>
    <th class="tg-c3ow">SSIM</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" colspan="9">Task-specific models</td>
  </tr>
  <tr>
    <th class="tg-nrix" rowspan="4">ViT-Large</th>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">39.74</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">0.969</span></td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">29.95</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">0.879</span></td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">18.91</span></td>
    <td class="tg-c3ow">0.741</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">27.51</span></td>
    <td class="tg-c3ow">0.882</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="9">Universal models</td>
  </tr>
  <tr>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">ViT-Large</span></td>
    <td class="tg-c3ow">39.28</td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">0.967</span></td>
    <td class="tg-c3ow">30.75</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.893</span></td>
    <td class="tg-c3ow">21.69</td>
    <td class="tg-c3ow">0.850</td>
    <td class="tg-c3ow">20.57</td>
    <td class="tg-c3ow">0.680</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ProRes</td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">39.28</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.967</span></td>
    <td class="tg-c3ow">30.67</td>
    <td class="tg-c3ow">0.891</td>
    <td class="tg-c3ow">22.73</td>
    <td class="tg-c3ow">0.877</td>
    <td class="tg-c3ow">28.03</td>
    <td class="tg-c3ow">0.897</td>
  </tr>
</tbody>
</table>

--- -->
<!-- 
#### Different training strategies for ProRes with degradation-aware visual prompts:

<table class="tg">
<thead>
  <tr>
    <th class="tg-nrix" colspan="2" rowspan="2">Prompt</th>
    <th class="tg-c3ow" colspan="2">denoising</th>
    <th class="tg-c3ow" colspan="2">deraining</th>
    <th class="tg-c3ow" colspan="2">enhance</th>
    <th class="tg-c3ow" colspan="2">deblurring</th>
  </tr>
  <tr>
    <th class="tg-c3ow" colspan="2">SIDD</th>
    <th class="tg-c3ow" colspan="2">5 datasets</th>
    <th class="tg-c3ow" colspan="2">LoL</th>
    <th class="tg-c3ow" colspan="2">4 datasets</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh"><span style="font-weight:normal;font-style:normal">Initialization</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">Learnable</span></td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
    <td class="tg-c3ow">PSNR</td>
    <td class="tg-c3ow">SSIM</td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:normal;font-style:normal">Random</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">Learnable</span></td>
    <td class="tg-c3ow">39.24</td>
    <td class="tg-c3ow">0.966</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">29.98</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.881</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">10.60</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.417</span></td>
    <td class="tg-c3ow">26.19</td>
    <td class="tg-c3ow">0.844</td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:normal;font-style:normal">Random</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">Detached</span></td>
    <td class="tg-c3ow">39.14</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.966</span></td>
    <td class="tg-c3ow">29.98</td>
    <td class="tg-c3ow">0.877</td>
    <td class="tg-c3ow">22.02</td>
    <td class="tg-c3ow">0.819</td>
    <td class="tg-c3ow">28.10</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.898</span></td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:normal;font-style:normal">Pre-trained</span></td>
    <td class="tg-c3ow">Learnable</td>
    <td class="tg-c3ow">39.26</td>
    <td class="tg-c3ow">0.967</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">30.20</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.884</span></td>
    <td class="tg-c3ow">22.47</td>
    <td class="tg-c3ow">0.876</td>
    <td class="tg-c3ow">27.83</td>
    <td class="tg-c3ow">0.891</td>
  </tr>
  <tr>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal;text-decoration:none">Pre-trained</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">Detached</span></td>
    <td class="tg-c3ow"><span style="font-weight:normal;font-style:normal">39.28</span></td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal;text-decoration:none">0.967</span></td>
    <td class="tg-c3ow">30.67</td>
    <td class="tg-c3ow">0.891</td>
    <td class="tg-c3ow">22.73</td>
    <td class="tg-c3ow">0.877</td>
    <td class="tg-c3ow">28.03</td>
    <td class="tg-c3ow">0.897</td>
  </tr>
</tbody>
</table>
  
--- -->


<!-- #### Prompt tuning on the FiveK and RESIDE-6K datasets:

<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="3">Methods</th>
    <th class="tg-9wq8" colspan="2">Enhancement</th>
    <th class="tg-9wq8" colspan="2">Dehazing</th>
  </tr>
  <tr>
    <th class="tg-9wq8" colspan="2">FiveK</th>
    <th class="tg-9wq8" colspan="2">RESIDE-6K</th>
  </tr>
  <tr>
    <th class="tg-9wq8">PSNR</th>
    <th class="tg-9wq8">SSIM</th>
    <th class="tg-9wq8">PSNR</th>
    <th class="tg-9wq8">SSIM</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal;text-decoration:none">ProRes </span>w/o Prompt Tuning</td>
    <td class="tg-9wq8">18.94</td>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal;text-decoration:none">0.815</span></td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">-</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><span style="font-weight:400;font-style:normal;text-decoration:none">ProRes w/ Prompt Tuning</span></td>
    <td class="tg-9wq8">22.78</td>
    <td class="tg-9wq8">0.839</td>
    <td class="tg-9wq8">21.47</td>
    <td class="tg-9wq8">0.840</td>
  </tr>
</tbody>
</table> -->

<!--
## Visualizations

### Control Ability
#### 1.Independent Control
Visualization results processed from images of different corruptions. Compared with the original inputs, the outputs are consistent with the given visual prompts.
![](figures/S1_independent.jpg)

#### 2. Sensitive to Irrelevant Task-specific Prompts
Visualization results processed by different prompts. Compared with the original inputs, the outputs remain unchanged with irrelevant visual prompts.
![](figures/S2_irrelevant.jpg)

#### 3. Tackle Complicated Corruptions
Visualization results processed by ProRes from images of mixed types of degradation, i.e., low-light and rainy. ProRes adopts two visual prompts for low-light enhancement (E) and deraining (D) and combines the two visual prompts by linear weighted sum, i.e., αD + (1 − α)E, to control the restoration process.
![](figures/S3_combine.jpg)


### Adaptation on New Datasets & Task
#### 1. Low-light Enhancement Results
Visualization results of ProRes on the FiveK dataset. We adopt two settings, i.e., direct inference and prompt tuning, to evaluate ProRes on the FiveK dataset (a new dataset for low-light enhancement).
![](figures/tuning_fivek.jpg)
#### 2. Dehazing Results
Visualization results of ProRes on the RESIDE-6K dataset via prompt tuning for image dehazing (a new task).
![](figures/tuning_reside.jpg)
-->



## Acknowledgement
This project is based on [PromptIR](https://github.com/va1shn9v/PromptIR), [AdaIR](https://github.com/c-yn/AdaIR), [Perceive-IR](https://github.com/House-yuyu/Perceive-IR) and [ProRes](https://github.com/leonmakise/ProRes). Thanks for their wonderful work!
