

# Finetune BLIP2 on ImageNet using 4 GPUs

1. Download `https://github.com/salesforce/LAVIS` and cd to the directory `cd LAVIS`
2. Install its Python environment `lavis` and activate it `conda activate lavis`
3. Run `prepare_data.py`, will generate `imagenet_lavis_train.json` and `imagenet_lavis_val.json`
4. Run `cp imagenet_lavis_train.json /home/.cache/lavis/coco/annotations/coco_karpathy_train.json` and `cp imagenet_lavis_val.json /home/.cache/lavis/coco/annotations/coco_karpathy_val.json`
5. Train model `bash train.sh`
6. Evaluate model `eval.ipynb`