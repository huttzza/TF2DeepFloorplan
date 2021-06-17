* [fork & modify TF2DeepFloorplan](#fork-&-modify-TF2DeepFloorplan)
* [README.md of original TF2DeepFloorplan](#TF2DeepFloorplan)
* [original research repository](https://github.com/zlzeng/DeepFloorplan)

# fork & modify TF2DeepFloorplan
이 repository는 [DeepFloorplan](https://github.com/zlzeng/DeepFloorplan)의 tensorflow, python 버전이 향상된 repository이다.

Network 구조에 대한 자세한 내용은 [논문](resources/Deep Floor Plan Recognition Using a Multi-Task Network with Room-Boundary-Guided Attention.pdf)

도면 이미지와 아키스케치로 편집한 벽 데이터를 활용하여 해당 모델의 재학습을 진행하였다.

2021.04.30 - 06.15

> **수정사항**
> ---
> * 추가 dataset에 대한 재학습 가능
> * dataset 추가 시 tfrecord 자동 생성
> * 실행 시 tfrecord 존재여부 확인 및 자동 생성
> * acc 확인 코드 삽입

## Network
> ### **FCN(Fully Convolutional Network) 사용**
> ---
> : VGG-16 + decoding layer
> 
> - VGG-16의 마지막 Fully Connected Layer에서 classification의 위치 정보가 사라진다.
>
>   (모든 노드들이 서로 곱하고 더해진다)
>
> - FCL을 제거하고, FCN으로 대치한다.
> - VGG를 통해 줄어든 feature의 크기를 입력 사이즈에 맞게 늘이기 위해서는 decoding 작업이 필요
> - room boundary, room type에 따라 서로 다른 decoding 작업이 사용된다.
> - 논문에서 제안된 decoding layer는 기존의 segentation 알고리즘(DeepLab, PSPNet)보다 더 나은 성능을 보인다고 설명하고 있다.

## 실행
```
python train.py
```

재학습을 시키고 싶다면, 기존의 dataset과 같은 형태의 새로운 dataset을 생성하여 `dataset/train`폴더에 넣은 후, `newDS` 파라미터를 `True`로 설정한 후 실행시키면 된다
```
python train.py --newDS=True --cmap=31
```

`cmap` 파라미터를 통해 room_type의 class를 지정해줄 수도 있다.

---

# TF2DeepFloorplan [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
This repo contains a basic procedure to train the DNN model suggested by the paper ['Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention'](https://arxiv.org/abs/1908.11025). It rewrites the original codes from [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan) into newer versions of Tensorflow and Python. 
<br>
Network Architectures from the paper, <br>
<img src="resources/dfpmodel.png" width="50%"><img src="resources/features.png" width="50%">

## Requirements
Install the packages stated in `requirements.txt`, including `matplotlib`,`numpy`,`opencv-python`,`pdbpp`, `tensorflow-gpu` and `tensorboard`. <br>
The code has been tested under the environment of Python 3.7.4 with tensorflow-gpu==2.3.0, cudnn==7.6.5 and cuda10.1_0. Used Nvidia RTX2080-Ti eGPU, 60 epochs take approximately 1 hour to complete.

## How to run?
1. Install packages via `pip` and `requirements.txt`.
```
pip install -r requirements.txt
```
2. According to the original repo, please download r3d dataset and transform it to tfrecords `r3d.tfrecords`.
3. Run the `train.py` file  to initiate the training, 
```
python train.py [--batchsize 2][--lr 1e-4][--epochs 1000]
[--logdir 'log/store'][--saveTensorInterval 10][--saveModelInterval 20]
```
- for example,
```
python train.py --batchsize=8 --lr=1e-4 --epochs=60 --logdir=log/store
```
4. Run Tensorboard to view the progress of loss and images via,
```
tensorboard --logdir=log/store
```

## Result
The following top left figure illustrates the result of the training image after 60 epochs, the first row is the ground truth (left:input, middle:boundary, right:room-type), the second row is the generated results. However, the result is not yet postprocessed, so the colors do not represent the classes, the edges are not smooth and the same area does not only show one class. <br>
The remaining figures are the graphs of total loss, loss for boundary and loss for room.
<img src="resources/epoch60.png" width="40%">
<img src="resources/Loss.png" width="40%">
<img src="resources/LossB.png" width="40%">
<img src="resources/LossR.png" width="40%">
