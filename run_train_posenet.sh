PWD=$(pwd)
mkdir $PWD/checkpoints/
EXPNAME=posenet
CHECKPOINT_DIR=$PWD/checkpoints/$EXPNAME
mkdir $CHECKPOINT_DIR
DATAROOT_DIR=$PWD/data_kitti
CUDA_VISIBLE_DEVICES=0 python src/train_main_posenet.py --dataroot $DATAROOT_DIR\
 --checkpoints_dir $CHECKPOINT_DIR --which_epoch -1 --save_latest_freq 1000\
  --batchSize 1 --display_freq 50 --name $EXPNAME --lambda_S 0.01 --smooth_term 2nd --use_ssim --display_port 8009
