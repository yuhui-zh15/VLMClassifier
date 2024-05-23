# Open World

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llava7b.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-13b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llava13b.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-v1.6-vicuna-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llavanext7bvicuna.jsonl --including_label False --batch_size 1
python main.py --method vlm --model_id llava-hf/llava-v1.6-mistral-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llavanext7bmistral.jsonl --including_label False --batch_size 1
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_blip2.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_iblip7b.jsonl --including_label False --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-13b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_iblip13b.jsonl --including_label False --batch_size 8

# Close World

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llava7b_196classes.jsonl --including_label True --n_labels 196 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-13b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llava13b_196classes.jsonl --including_label True --n_labels 196 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-v1.6-vicuna-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llavanext7bvicuna_196classes.jsonl --including_label True --n_labels 196 --batch_size 1
python main.py --method vlm --model_id llava-hf/llava-v1.6-mistral-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llavanext7bmistral_196classes.jsonl --including_label True --n_labels 196 --batch_size 1
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_blip2_196classes.jsonl --including_label True --n_labels 196 --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_iblip7b_196classes.jsonl --including_label True --n_labels 196 --batch_size 8
python main.py --method vlm --model_id Salesforce/instructblip-vicuna-13b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_iblip13b_196classes.jsonl --including_label True --n_labels 196 --batch_size 8

# Baselines

python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_clipvitl336_196classes.jsonl --including_label False --batch_size 32
python main.py --method clip --model_id EVA01-CLIP-g-14-plus --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_evaclipg_196classes.jsonl --including_label False --batch_size 32
