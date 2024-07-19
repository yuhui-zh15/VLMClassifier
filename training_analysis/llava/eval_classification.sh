python -m llava.eval.model_vqa \
    --model-path ./checkpoints/processed-llava-v1.5-7b-imagenet/ \
    --question-file ./playground/data/imagenet_llava_val.jsonl \
    --image-folder "" \
    --answers-file ./playground/data/final_outputs/imagenet_llava_val_finetuned_1epochs.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/processed-llava-v1.5-7b-flowers-100epochs/ \
    --question-file ./playground/data/flowers_llava_val.jsonl \
    --image-folder "" \
    --answers-file ./playground/data/final_outputs/flowers_llava_val_finetuned_100epochs.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/processed-llava-v1.5-7b-cars-100epochs/ \
    --question-file ./playground/data/cars_llava_val.jsonl \
    --image-folder "" \
    --answers-file ./playground/data/final_outputs/cars_llava_val_finetuned_100epochs.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/processed-llava-v1.5-7b-caltech-100epochs/ \
    --question-file ./playground/data/caltech_llava_val.jsonl \
    --image-folder "" \
    --answers-file ./playground/data/final_outputs/caltech_llava_val_finetuned_100epochs.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/processed-llava-v1.5-7b-imagenetllava-1epochs/ \
    --question-file ./playground/data/imagenet_llava_val.jsonl \
    --image-folder "" \
    --answers-file ./playground/data/final_outputs/imagenet_llava_val_combinedfinetuned7b_1epochs.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
