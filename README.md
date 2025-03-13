# MSAFF
A Multi-Stage Adaptive Feature Fusion Neural Network for Multimodal Gait Recognition

The article has been accepted by IJCB2023 (oral).
- **IJCB 2023 Version**: [Download paper (IJCB2023)]()
- **ArXiv Preprint**: [Download preprint (ArXiv)](https://arxiv.org/pdf/2312.14410.pdf)
## Updates
Following the code open-sourcing, I’m grateful for the unexpected attention this early-stage work has received. Thank you all for your support. Addressing some reproduction issues raised by peers (likely due to dataset processing):
- Datasets: Processed versions of CASIA-B and Gait-3D are available [here](https://pan.baidu.com/s/1M8UepY3vWV4b_rIWwMNBxA?pwd=fcbs) code(fcbs).
- Gait-3D and GREW: Experiments on Gait-3D and GREW were modified from the early open-sourced OpenGait framework.
- CASIA-B: Original experiments on CASIA-B were modified from the early open-sourced GaitSet framework. Code and weights for verification are now provided [here](https://pan.baidu.com/s/1M8UepY3vWV4b_rIWwMNBxA?pwd=fcbs) code(fcbs).
- Training Stability: Both GaitSet and early OpenGait frameworks exhibit training instability in some algorithms (e.g., accuracy fluctuations of ±1-2% per training ), though the root cause remains unclear. To ensure optimal results, we trained each model ≥5 times and performed testing every 10 iterations during the final 20,000 iterations to select the best weights. This methodology produced the results reported in our paper.
- Future Plans: We will adapt this code to the CCGR and CCGR-MINI datasets and enhance maintenance and support for the CCGR series. Welcome to utilize CCGR series (https://github.com/ShinanZou/CCGR).
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
@INPROCEEDINGS{ShinanZouMSAFF
  author={Zou, Shinan and Xiong, Jianbo and Fan, Chao and Yu, Shiqi and Tang, Jin},
  booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)}, 
  title={A Multi-Stage Adaptive Feature Fusion Neural Network for Multimodal Gait Recognition}, 
  year={2023}}
```
## Acknowledgement

Here are some great resources we benefit:

- The codebase is based on [OpenGait](https://github.com/ShiqiYu/OpenGait) and [Gait3D](https://gait3d.github.io).
