python main_api.py --api claude --data_path ../data/imagewikiqa.jsonl --output_path ./outputs/imagewikiqa_claude.jsonl --threads 16
python main_api.py --api gpt4v --data_path ../data/imagewikiqa.jsonl --output_path ./outputs/imagewikiqa_gpt4v.jsonl --threads 16
python main_api.py --api gemini --data_path ../data/imagewikiqa.jsonl --output_path ./outputs/imagewikiqa_gemini.jsonl --threads 16
