# MSAFF
A Multi-Stage Adaptive Feature Fusion Neural Network for Multimodal Gait Recognition

The article has been accepted by IJCB2023
## Requirements

- pytorch >= 1.6
- torchvision
- pyyaml
- tensorboard
- opencv-python
- tqdm
- py7zr
- tabulate
- termcolor

### Installation

You can replace the second command from the bottom to install
[pytorch](https://pytorch.org/get-started/previous-versions/#v110) 
based on your CUDA version.

```
git clone https://github.com/ShinanZou/MSAFF.git
cd MSAFF
conda create --name py37torch160 python=3.7
conda activate py37torch160
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install tqdm pyyaml tensorboard opencv-python tqdm py7zr tabulate termcolor
```

### Data Pretreatment

Run the following command to preprocess the CASIA-B and Gait3D dataset.

```
python misc/pretreatment.py --input_path '2D_Silhouettes' --output_path 'sils-64-44-pkl' --img_h 64 --img_w 44
python misc/pretreatment_ske.py --input_path '2D_Poses' --output_path 'skes-pkl'
```

Run the following command to preprocess the GREW dataset.

```
// silhouettes
python misc/pretreatment_grew.py --input_path "GREW" --output_path "GREW-64-44-pkl" --img_h 64 --img_w 44 --subset "train"
python misc/pretreatment_grew.py --input_path "GREW" --output_path "GREW-64-44-pkl" --img_h 64 --img_w 44 --subset "test/gallery"
python misc/pretreatment_grew_probe.py --input_path "GREW" --output_path "GREW-64-44-pkl" --img_h 64 --img_w 44

// skeletons
python misc/pretreatment_grew_ske.py --input_path "GREW" --output_path "GREW-skes-pkl" --img_h 64 --img_w 44 --subset "train"
python misc/pretreatment_grew_ske.py --input_path "GREW" --output_path "GREW-skes-pkl" --img_h 64 --img_w 44 --subset "test/gallery"
python misc/pretreatment_grew_ske_probe.py --input_path "GREW" --output_path "GREW-skes-pkl" --img_h 64 --img_w 44
```

## Train

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/MsaffGait_CasiaB.yaml --phase train
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/MsaffGait_Gait3D.yaml --phase train
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/MsaffGait_GREW.yaml --phase train
```

## Test

Run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/MsaffGait_CasiaB.yaml --phase test
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/MsaffGait_Gait3D.yaml --phase test
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 lib/main.py --cfgs ./config/MsaffGait_GREW.yaml --phase test
```
## Citation

Please cite this paper in your publications if it helps your research:

```BibTeX

```
## Acknowledgement

Here are some great resources we benefit:

- The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait) and [Gait3D](https://gait3d.github.io).
