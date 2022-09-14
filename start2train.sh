
#python train.py --dataset cityscapes_preprocessed --split cityscapes_preprocessed --scheduler_step_size 14  --batch 16  --model_name mono_model --png --data_path data_path/cityscapes_preprocessed
CUDA_VISIBLE_DEVICES=0 python train.py --scheduler_step_size 14  --batch 12  --model_name aug-depth --png --data_path ../../datasets/raw_data --load_weights_folder /home/inspur/MAX_SPACE/yangli/newidea-newbackbobe/pretrained-model/diffnet_640x192
