# Open World

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llava7b.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-13b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llava13b.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-v1.6-vicuna-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llavanext7bvicuna.jsonl --including_label False --batch_size 1
python main.py --method vlm --model_id llava-hf/llava-v1.6-mistral-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llavanext7bmistral.jsonl --including_label False --batch_size 1
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_blip2.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_iblip7b.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-13b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_iblip13b.jsonl --including_label False --batch_size 8

# Close World

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llava7b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-13b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llava13b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-v1.6-vicuna-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llavanext7bvicuna_100classes.jsonl --including_label True --n_labels 100 --batch_size 1
python main.py --method vlm --model_id llava-hf/llava-v1.6-mistral-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llavanext7bmistral_100classes.jsonl --including_label True --n_labels 100 --batch_size 1
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_blip2_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_iblip7b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-13b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_iblip13b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8

# Baselines

python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_clipvitl336_100classes.jsonl --including_label False --batch_size 32
python main.py --method clip --model_id EVA01-CLIP-g-14-plus --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_evaclipg_100classes.jsonl --including_label False --batch_size 32
