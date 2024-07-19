python -m llava.eval.model_vqa \
    --model-path /home/VlmClassifier/training_analysis/LLaVA/checkpoints/processed-llava-v1.5-7b-imagenet-and-llava \
    --question-file /home/VlmClassifier/data/imagenetqa.jsonl  \
    --image-folder "" \
    --answers-file ./playground/data/imagewikiqa_predictions_llava-7b_imagenet-and-llava-trained.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

