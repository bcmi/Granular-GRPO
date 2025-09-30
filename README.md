# Granular-GRPO

<details><summary>Click for the full abstract of Light-A-Video</summary>

> The integration of online reinforcement learning (RL) into diffusion and flow models has recently emerged as a promising approach for aligning generative models with human preferences. Stochastic sampling via Stochastic Differential Equations (SDE) is employed during the denoising process
to generate diverse denoising directions for RL exploration. While existing methods effectively explore potential high-value samples,
they suffer from sub-optimal preference alignment due to sparse and narrow reward signals. To address these challenges, we propose a novel \textbf{G}ranular-\textbf{GRPO} ($\text{G}^2$RPO ) framework that achieves precise and comprehensive reward assessments of sampling directions in reinforcement learning of flow models. Specifically, a \textit{Singular Stochastic Sampling} strategy is introduced to support step-wise stochastic exploration 
while enforcing a high correlation between the reward and the injected noise, thereby facilitating a faithful reward for each SDE perturbation.
Concurrently, to eliminate the bias inherent in fixed-granularity denoising, we introduce a \textit{Multi-Granularity Advantage Integration} module 
that aggregates advantages computed at multiple diffusion scales, producing a more comprehensive and robust evaluation of the sampling directions.
Experiments conducted on various reward models, including both in-domain and out-of-domain evaluations, demonstrate that our $\text{G}^2$RPO significantly outperforms existing flow-based GRPO baselines, highlighting its effectiveness and robustness.
</details>

**[$\text{G}^2$RPO: Granular GRPO for Precise Reward in Flow Models]()** 
</br>
[Yujie Zhou*](https://github.com/YujieOuO/),
[Pengyang Ling*](https://github.com/LPengYang/),
[Jiazi Bu*](https://github.com/Bujiazi/),
[Yibin Wang](https://codegoat24.github.io/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Jiaqi Wang<sup>†</sup>](https://myownskyw7.github.io/),
[Li Niu<sup>†</sup>](https://www.ustcnewly.com/)
[Guangtao Zhai](https://faculty.sjtu.edu.cn/zhaiguangtao/en/index.htm/),
(*Equal Contribution)(<sup>†</sup>Corresponding Author)

[![arXiv](https://img.shields.io/badge/arXiv-2502.08590-b31b1b.svg)](https://github.com/bcmi/Granular-GRPO)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://github.com/bcmi/Granular-GRPO)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-red)](https://github.com/bcmi/Granular-GRPO)

## 📜 News

**[2025/9/30]** The paper and project page are released!

## 🏗️ Todo
- [] Release a gradio demo.

## 📚 Gallery
We show more results in the [Project Page](https://github.com/bcmi/Granular-GRPO).

## 🚀 Method Overview

<div align="center">
    <img src='__assets__/g2rpo.png'/>
</div>

## 🔧 Installations

### Setup repository and conda environment

```bash
git clone https://github.com/bcmi/Granular-GRPO.git
cd Granular-GRPO

conda create -n g2rpo python=3.10
conda activate g2rpo

bash env_setup.sh

git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e . 
cd ..
```

The environment dependency is the same as [DanceGRPO](https://github.com/XueZeyue/DanceGRPO).

## 🔑 Model Preparations

### 1. FLUX

### 2. Reward Models

#### HPS-v2.1

#### CLIP Score

## 🎈 Quick Start

### Preprocess Data

### Run Training

### Run Inference

### Run Evaluation

## 💞 Acknowledgement
The code is built upon the below repositories, we thank all the contributors for open-sourcing.

* [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
* [Flow-GRPO](https://github.com/yifan123/flow_grpo)
* [FastVideo](https://github.com/hao-ai-lab/FastVideo)
* [DDPO](https://github.com/kvablack/ddpo-pytorch)