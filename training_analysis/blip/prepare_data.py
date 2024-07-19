import json
import random

dataset = "imagenet"

data = [json.loads(line) for line in open(f"../../data/{dataset}.jsonl")]
data = [item for item in data if item["split"] == "train"]
random.seed(1234)
random.shuffle(data)

new_data = []
for idx, item in enumerate(data):
    new_item = {
        "image_id": f'{idx}-{item["label"]}',
        "image": item["image"],
        "caption": f'{item["label"]}',
    }
    new_data.append(new_item)
json.dump(new_data, open(f"{dataset}_lavis_train.json", "w"), indent=2)

data = [json.loads(line) for line in open(f"../../data/{dataset}.jsonl")]
data = [item for item in data if item["split"] == "test"]
random.seed(1234)
random.shuffle(data)

new_data = []
for idx, item in enumerate(data):
    new_item = {
        "image_id": f'{idx}-{item["label"]}',
        "image": item["image"],
        "caption": [f'{item["label"]}'],
    }
    new_data.append(new_item)
json.dump(new_data, open(f"{dataset}_lavis_val.json", "w"), indent=2)
