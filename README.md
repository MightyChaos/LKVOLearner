# LKVOLearner

## Prerequisite
PyTorch 0.3.0

pip install visdom
pip install dominate



## Dataset preparation
### 1. download KITTI dataset
```
mkdir kitti
cd kitti
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip
unzip raw_data_downloader.zip
bash raw_data_downloader.sh
```

### 2. preprocessing
```
python data/prepare_train_data.py --dataset_dir=/path/to/raw/kitti/dataset/ --dataset_name='kitti_raw_eigen' --dump_root=/path/to/resulting/formatted/data/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```

## train from scratch with PoseNet
```
PWD='pwd'
mkdir $PWD/checkpoints/
EXPNAME=posenet
CHECKPOINT_DIR=$PWD/checkpoints/$EXPNAME
mkdir checkpoints_dir
DATAROOT_DIR=$PWD/data_kitti
CUDA_VISIBLE_DEVICES=0 python3 train_main_posenet.py --dataroot $DATAROOT_DIR --checkpoints_dir $CHECKPOINT_DIR --which_epoch -1 --save_latest_freq 1000 --batchSize 1 --display_freq 50 --name $EXPNAME --lambda_S 0.01 --smooth_term 2nd --use_ssim
```
## train from scratch with DDVO
```
CUDA_VISIBLE_DEVICES=0 python3 train_main.py --dataroot /dir-to-dataset/ --checkpoints_dir /dir-to-checkpoints/ --which_epoch -1 --save_latest_freq 1000 --batchSize 1 --display_freq 50 --name DDVO --lambda_S 0.01 --smooth_term 2nd --use_ssim
```
