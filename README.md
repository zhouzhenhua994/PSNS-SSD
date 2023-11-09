

# “PSNS-SSD: Pixel-level Suppress Non-Salient Semantic and Multicoupled Channel Enhancement Attention for 3D object detection”



## Getting Started

### Installation

a. Clone this repository
```shell
git clone https://github.com/zhouzhenhua994/PSNS-SSD
```
b. Configure the environment

We have tested this project with the following environments:
* Ubuntu20.04
* Python = 3.8
* PyTorch = 1.7
* CUDA = 11.3
* CMake >= 3.13
* spconv = 2.2

c. Install `pcdet` toolbox.
```shell
pip install -r requirements.txt
python setup.py develop
```

d. Prepare the datasets. 

Download the official KITTI with [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) and Waymo datasets, then organize the unzipped files as follows:
```
PSNS-SSD
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   ├── testing
│   │   ├── calib & velodyne & image_2
├── pcdet
├── tools
```
Generate the data infos by running the following commands:
```python 
# KITTI dataset
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```


### Quick Inference
We provide the pre-trained weight file so you can just run with that:
```shell
cd tools 
# To achieve fully GPU memory footprint (NVIDIA RTX3090, 24GB).
python test.py --cfg_file cfgs/kitti_models/PSNS-SSD.yaml --batch_size 80 \
    --ckpt PSNS-SSD.pth
```



### Training
```shell
python train.py --cfg_file cfgs/kitti_models/PSNS-SSD.yaml
```


### Evaluation

Evaluate with single or multiple GPUs: (e.g., KITTI dataset)
```shell
python test.py --cfg_file cfgs/kitti_models/PSNS-SSD.yaml  --batch_size ${BATCH_SIZE} --ckpt ${PTH_FILE}
```
