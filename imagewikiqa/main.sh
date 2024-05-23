python main.py --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_llava7b.jsonl --batch_size 8
python main.py --model_id llava-hf/llava-1.5-13b-hf --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_llava13b.jsonl --batch_size 8
python main.py --model_id llava-hf/llava-v1.6-vicuna-7b-hf --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_llavanext7bvicuna.jsonl --batch_size 1
python main.py --model_id llava-hf/llava-v1.6-mistral-7b-hf --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_llavanext7bmistral.jsonl --batch_size 1
python main.py --model_id Salesforce/blip2-opt-2.7b --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_blip2.jsonl --batch_size 8
python main.py --model_id Salesforce/instructblip-vicuna-7b --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_iblip7b.jsonl --batch_size 8
python main.py --model_id Salesforce/instructblip-vicuna-13b --data_path ../data/imagewikiqa.jsonl --output_path outputs/imagewikiqa_iblip13b.jsonl --batch_size 8
