# LKVOLearner

## train from scratch with DDVO
CUDA_VISIBLE_DEVICES=0 python3 train_main.py --dataroot /dir-to-dataset/ --checkpoints_dir /dir-to-checkpoints/ --which_epoch -1 --save_latest_freq 1000 --batchSize 1 --display_freq 50 --name DDVO --lambda_S 0.01 --smooth_term 2nd --use_ssim
