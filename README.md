# Cancer Site Classification

This is a small example for you to get started with your course. 

In this model, we use <a href="https://arxiv.org/abs/1512.00567"> Inception-V3 </a> as feature extractor. Extracted tiles that contains tissues are feeded to Inception-V3. The representations are feeded to a classifier. Here we aim to classify two types of cancer sites(COAD and UCEC). 

<p align='center'>  
  <img src='fig/BaselineModel.png' width='600' height='200' />
</p>

## Prerequisites
- python==3.6.3
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
cd CancerSiteClassification/
```
- download the <a href="https://arxiv.org/abs/1512.00567"> toy dataset </a>

The space would be about 44G. 

I prepared a small dataset. you can download <a href="https://arxiv.org/abs/1803.04054"> this toy dataset </a> and put it in rep folder. You can also try some other dataset.

### Extract tiles
Run the scripts to extract all the tiles contain tissues from the slides. A typical tile size is 1000X1000. All these tiles are resized to 299X299 which is used by <a href="https://arxiv.org/abs/1512.00567"> Inception-V3 </a>. Here we simply calculate the intensity of all the grayscale pixel values. If the intensity of a tile is blew the threshold, we keep the tile. 

```bash
cd gen_feature/
python gen_tile.py
```

### Train model
Go to model folder and run the bash scripts. Be sure to specify the GPU you want to run the model on.
```bash
cd model/
sh run.sh
```

