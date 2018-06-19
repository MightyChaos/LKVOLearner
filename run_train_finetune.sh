MODEL_ID=9
PWD=$(pwd)
mkdir $PWD/checkpoints/
EXPNAME=finetune
CHECKPOINT_DIR=$PWD/checkpoints/$EXPNAME
mkdir $CHECKPOINT_DIR
# copy learnt model from PoseNet
POSENET_CKPT_DIR=$PWD/checkpoints/posenet

cp $(printf "%s/%s_pose_net.pth" "$POSENET_CKPT_DIR" "$MODEL_ID") $CHECKPOINT_DIR/pose_net.pth
cp $(printf "%s/%s_depth_net.pth" "$POSENET_CKPT_DIR" "$MODEL_ID") $CHECKPOINT_DIR/depth_net.pth

DATAROOT_DIR=$PWD/data_kitti

CUDA_VISIBLE_DEVICES=0 python src/train_main_finetune.py --dataroot $DATAROOT_DIR\
 --checkpoints_dir $CHECKPOINT_DIR --which_epoch -1 --save_latest_freq 1000\
  --batchSize 1 --display_freq 50 --name $EXPNAME\
  --lk_level 1 --lambda_S 0.01 --smooth_term 2nd --use_ssim --display_port 8009 --epoch_num 10
  --lr 0.00001
  
  
