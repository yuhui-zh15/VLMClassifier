#--------------#
# The folder structure should be organized as follows before training.
# VLMClassifierAnalysis
# |-- train_analysis
#    |-- dev_output
#        |-- {dataset}
#        |-- {dataset}_feature
#            |-- train
#            |-- test
#        |-- {dataset}_feature_eva
#            |-- train
#            |-- test
# Notice that if you want to tune the entire CLIP vision encoder, 
# you need to tune linear only first to get the pretrained classifier weight.
# You also need to export 2 gpu to gain enough memory, or you can reduce the batchsize.
# If you want to use EVA-CLIP, please 'conda activate rei' first, and run 'python train_analysis/finetune_clip_eva.py'.
# EVA-CLIP only supports trian linear head.
#--------------#

export CUDA_VISIBLE_DEVICES=0,1
python train_analysis/finetune_clip_eva.py  --dataset imagenet --epochs 40 --model_id EVA01-CLIP-g-14-plus
python train_analysis/finetune_clip_eva.py  --dataset flowers --epochs 300 --model_id EVA01-CLIP-g-14-plus
python train_analysis/finetune_clip_eva.py  --dataset cars --epochs 300 --model_id EVA01-CLIP-g-14-plus
python train_analysis/finetune_clip_eva.py  --dataset caltech --epochs 300 --model_id EVA01-CLIP-g-14-plus
python train_analysis/finetune_clip.py  --dataset imagenet --epochs 40 --param_to_ft linear
python train_analysis/finetune_clip.py  --dataset flowers --epochs 300 --param_to_ft linear
python train_analysis/finetune_clip.py  --dataset cars --epochs 300 --param_to_ft linear
python train_analysis/finetune_clip.py  --dataset caltech --epochs 300 --param_to_ft linear
python train_analysis/finetune_clip.py  --dataset imagenet --epochs 2 --param_to_ft whole
python train_analysis/finetune_clip.py  --dataset caltech --epochs 10 --param_to_ft whole
python train_analysis/finetune_clip.py  --dataset cars --epochs 20 --param_to_ft whole
python train_analysis/finetune_clip.py  --dataset flowers --epochs 20 --param_to_ft whole