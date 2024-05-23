

python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path outputs/n_classes/imagenet_blip2_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path outputs/n_classes/imagenet_blip2_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path outputs/n_classes/imagenet_blip2_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path outputs/n_classes/imagenet_blip2_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_llava7b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_llava7b_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_llava7b_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_llava7b_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_blip2_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_blip2_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_blip2_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_blip2_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_clipvitl336_100classes.jsonl --including_label True --n_labels 100 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_clipvitl336_20classes.jsonl --including_label True --n_labels 20 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_clipvitl336_5classes.jsonl --including_label True --n_labels 5 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/n_classes/flowers_clipvitl336_2classes.jsonl --including_label True --n_labels 2 --batch_size 32

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_llava7b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_llava7b_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_llava7b_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_llava7b_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_blip2_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_blip2_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_blip2_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_blip2_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_clipvitl336_100classes.jsonl --including_label True --n_labels 100 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_clipvitl336_20classes.jsonl --including_label True --n_labels 20 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_clipvitl336_5classes.jsonl --including_label True --n_labels 5 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/n_classes/cars_clipvitl336_2classes.jsonl --including_label True --n_labels 2 --batch_size 32

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_llava7b_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_llava7b_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_llava7b_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_llava7b_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_blip2_100classes.jsonl --including_label True --n_labels 100 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_blip2_20classes.jsonl --including_label True --n_labels 20 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_blip2_5classes.jsonl --including_label True --n_labels 5 --batch_size 8
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_blip2_2classes.jsonl --including_label True --n_labels 2 --batch_size 8

python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_clipvitl336_100classes.jsonl --including_label True --n_labels 100 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_clipvitl336_20classes.jsonl --including_label True --n_labels 20 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_clipvitl336_5classes.jsonl --including_label True --n_labels 5 --batch_size 32
python main.py --method clip --model_id ViT-L/14@336px --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/n_classes/caltech_clipvitl336_2classes.jsonl --including_label True --n_labels 2 --batch_size 32