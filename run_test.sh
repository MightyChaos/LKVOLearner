DATAROOT=../kitti/
PWD=$(pwd)
CKPT=$PWD/checkpoints/finetune/5_model.pth
# CKPT=$PWD/posenet+ddvo.pth 
OUTPUT=$PWD/posenet+ddvo
CUDA_VISIBLE_DEVICES=0 nice -10 python src/testKITTI.py --dataset_root $DATAROOT --ckpt_file $CKPT --output_path $OUTPUT --test_file_list test_files_eigen.txt

python2 ~/SfMLearner/kitti_eval/eval_depth.py --kitti_dir=$DATAROOT --pred_file=$OUTPUT.npy --test_file_list test_files_eigen.txt