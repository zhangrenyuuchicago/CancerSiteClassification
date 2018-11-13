# Cancer Site Classification

This is a small example for you to get started with your course. 

<p align='center'>  
  <img src='fig/BaselineModel.png' width='600' height='280' />
</p>


## Prerequisites
tensorboard==1.11.0
tensorboardX==1.4
torch==0.4.1
torchvision==0.2.1
tqdm==4.19.5
openslide-python==1.1.1

## Getting Started

### extract tiles

cd gen_feature
python gen_tile.py

### train model
cd model
sh run.sh


