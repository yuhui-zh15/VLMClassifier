python main.py --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path outputs/imagenet_llava7b.jsonl
python main.py --model_id Salesforce/blip2-opt-2.7b --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path outputs/imagenet_blip2.jsonl

python main.py --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/flowers_llava7b.jsonl
python main.py --model_id Salesforce/blip2-opt-2.7b --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/flowers_blip2.jsonl

python main.py --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_llava7b.jsonl
python main.py --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/cars_blip2.jsonl

python main.py --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_llava7b.jsonl
python main.py --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/caltech_blip2.jsonl
