python main.py --api gpt4v --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path ./outputs/gpt4v_imagenet_ow.jsonl --including_label False
python main.py --api gpt4v --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path ./outputs/gpt4v_flowers_ow.jsonl --including_label False
python main.py --api gpt4v --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path ./outputs/gpt4v_cars_ow.jsonl --including_label False
python main.py --api gpt4v --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path ./outputs/gpt4v_caltech_ow.jsonl --including_label False

python main.py --api gemini --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path ./outputs/gemini_imagenet_ow.jsonl --including_label False
python main.py --api gemini --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path ./outputs/gemini_flowers_ow.jsonl --including_label False
python main.py --api gemini --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path ./outputs/gemini_cars_ow.jsonl --including_label False
python main.py --api gemini --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path ./outputs/gemini_caltech_ow.jsonl --including_label False

python main.py --api claude --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path ./outputs/claude_imagenet_ow.jsonl --including_label False
python main.py --api claude --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path ./outputs/claude_flowers_ow.jsonl --including_label False
python main.py --api claude --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path ./outputs/claude_cars_ow.jsonl --including_label False
python main.py --api claude --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path ./outputs/claude_caltech_ow.jsonl --including_label False





python main.py --api gpt4v --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path ./outputs/gpt4v_imagenet_cw.jsonl --including_label True --n_labels 998
python main.py --api gpt4v --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path ./outputs/gpt4v_flowers_cw.jsonl --including_label True --n_labels 102
python main.py --api gpt4v --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path ./outputs/gpt4v_cars_cw.jsonl --including_label True --n_labels 196
python main.py --api gpt4v --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path ./outputs/gpt4v_caltech_cw.jsonl --including_label True --n_labels 100

python main.py --api gemini --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path ./outputs/gemini_imagenet_cw.jsonl --including_label True --n_labels 998
python main.py --api gemini --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path ./outputs/gemini_flowers_cw.jsonl --including_label True --n_labels 102
python main.py --api gemini --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path ./outputs/gemini_cars_cw.jsonl --including_label True --n_labels 196
python main.py --api gemini --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path ./outputs/gemini_caltech_cw.jsonl --including_label True --n_labels 100

python main.py --api claude --data_path ../data/imagenet.jsonl --class_path ../data/imagenet_classes.json --split valid --output_path ./outputs/claude_imagenet_cw.jsonl --including_label True --n_labels 998
python main.py --api claude --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path ./outputs/claude_flowers_cw.jsonl --including_label True --n_labels 102
python main.py --api claude --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path ./outputs/claude_cars_cw.jsonl --including_label True --n_labels 196
python main.py --api claude --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path ./outputs/claude_caltech_cw.jsonl --including_label True --n_labels 100

