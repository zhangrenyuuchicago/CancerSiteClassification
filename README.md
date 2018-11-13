# Cancer Site Classification

This is a small example for you to get started with your course. 

<p align='center'>  
  <img src='fig/BaselineModel.png' width='600' height='200' />
</p>


## Prerequisites
- tensorboard==1.11.0
- tensorboardX==1.4
- torch==0.4.1
- torchvision==0.2.1
- tqdm==4.19.5
- openslide-python==1.1.1

## Getting Started

### Installation and Download
- install all required libs
- clone this rep
```bash
git clone https://github.com/zhangrenyuuchicago/CancerSiteClassification
cd CancerSiteClassification
```
- download the dataset

I prepared a small dataset. you can download it <a href="https://arxiv.org/abs/1803.04054"> and put it in rep folder. You can also try some other dataset.

### Extract tiles

```bash
cd gen_feature
python gen_tile.py
```

### Train model

```bash
cd model
sh run.sh
```

