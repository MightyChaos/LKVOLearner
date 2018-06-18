# Learning Depth from Monocular Videos using Direct Methods

Implementation of the methods in "[Learning Depth from Monocular Videos using Direct Methods](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf)".
If you find this code useful, please cite the paper:

```
@InProceedings{Wang_2018_CVPR,
author = {Wang, Chaoyang and Miguel Buenaposada, José and Zhu, Rui and Lucey, Simon},
title = {Learning Depth From Monocular Videos Using Direct Methods},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
## Dependencies
- Python 3.6
- PyTorch 0.3.1  (latter or eariler version of Pytorch is non-compatible.)

- visdom, dominate 


## Training
- ### data preparation
We refer "[SfMLeaner](https://github.com/tinghuiz/SfMLearner)" to prepare the training data from KITTI. We assume the processed data is put in "./data_kitti/" 

- ### Training with different pose prediction modules
start visdom server before for inspecting learning progress before starting the training process.
```
python -m visdom.server -port 8009
```
1. #### train from scratch with PoseNet
```
bash run_train_posenet.sh
```
see [run_train_posenet.sh](https://github.com/MightyChaos/LKVOLearner/blob/master/run_train_posenet.sh) for details.

2. #### finetune with DDVO
Use pretrained posenet to give initialization for DDVO. Corresponds to the results reported as "PoseNet+DDVO" in the paper.
```
bash run_train_finetune.sh
```
see [run_train_finetune.sh](https://github.com/MightyChaos/LKVOLearner/blob/master/run_train_finetune.sh) for details.

## Testing
- pretrained depth network reported as "Posenet-DDVO(KCS)" in the paper [[download](https://drive.google.com/file/d/1SJWLfA7kqpERj_U2gYXl7Vuy1eQyOO_K/view?usp=sharing)].
- depth prediction results on KITTI eigen test split:   [[Posenet(K)](https://drive.google.com/open?id=1Wj7ulSimrvrzNx4TRd-JspmX3DJwgPiV)], [[DDVO(K)](https://drive.google.com/open?id=1wiODwgX_Vm_w7fVK1y_X5CNJTtgaPwcN)], [[Posenet+DDVO(K)](https://drive.google.com/open?id=1uUQJLcUOoY2hG6QS_F-wbM3GDAjD-Z5h)],[[Posenet+DDVO(KCS)](https://drive.google.com/open?id=1hp4zFgK5NSNGdvaQL2ZumeinMQY_-AwK)]
## Acknowledgement
Part of the code structure is borrowed from "[Pytorch CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)"