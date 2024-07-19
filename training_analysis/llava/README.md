# Finetune LLaVA on ImageNet using 4 GPUs

1. Download `https://github.com/haotian-liu/LLaVA` and cd to the directory `cd LLaVA`
2. Install its Python environment `llava` and activate it `conda activate llava`
3. Replace two files `train.py` and `model_vqa.py` with the ones in this directory
4. Process data `prepare_data.ipynb`
5. Train model `finetune.sh` (train projector only) or `finetune_lora.sh` (train both projector and LLM with LoRA) or `finetune_imagenet_llava.sh` (train on both ImageNet and LLaVA instruction tuning data) or `finetune_caption.sh` (train on caption data, see Appendix)
6. Process trained model `process_model.ipynb`
7. Generate model predictions `eval_classification.sh` (classification tasks) or `eval_imagewikiqa.sh` (ImageWikiQA task)
8. Evaluate predictions `eval_classification.ipynb` (classification tasks) or `eval_imagewikiqa.ipynb` (ImageWikiQA task)