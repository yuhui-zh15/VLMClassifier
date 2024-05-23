from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
import torch
from PIL import Image
import json
from tqdm import trange
import random
import click
import os


def main(
    model_id, data_path, class_path, seed, output_path, batch_size, init_prompt=None
):
    if "llava-v1.6" in model_id:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "blip2" in model_id:
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    elif "instructblip" in model_id:
        processor = InstructBlipProcessor.from_pretrained(model_id)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    data = [json.loads(line) for line in open(data_path)]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    if os.path.exists(output_path):
        outputs = [json.loads(line) for line in open(output_path)]
        data = data[len(outputs) :]

    if init_prompt is None:
        init_prompt = "What type of object is in this photo?"

    all_features = []
    save_freq = 1024
    save_cnt = 0

    for i in trange(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        images = [Image.open(item["image"]) for item in batch]
        questions = [init_prompt for _ in batch]

        if "llava-v1.6-mistral" in model_id:
            prompts = [f"[INST] <image>\n{question}[/INST]" for question in questions]
        elif "llava-v1.6-vicuna" in model_id:
            prompts = [
                f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
                for question in questions
            ]
        elif "blip" in model_id:
            prompts = [f"Question: {question} Answer:" for question in questions]
        else:
            prompts = [
                f"USER: <image>\n{question}\nASSISTANT:" for question in questions
            ]

        inputs = processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            if "blip" in model_id:
                outputs = outputs.language_model_outputs
            hidden_states = outputs.hidden_states[-1]
            last_features = hidden_states[:, -1, :].unsqueeze(1)
            mean_features = hidden_states.mean(dim=1).unsqueeze(1)
            features = torch.cat([last_features, mean_features], dim=1)
        all_features.append(features.cpu())

        if i % save_freq == 0 or i == len(data) // batch_size * batch_size:
            all_features = torch.cat(all_features, dim=0)
            print(f"{all_features.shape=}")

            torch.save(all_features, f"{output_path}_{save_cnt}.pt")
            print(f"Saved features at {i}")

            save_cnt += 1
            all_features = []


@click.command()
@click.option("--model_id", default="llava-hf/llava-1.5-7b-hf")
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--class_path", default="../data/imagenet_classes.json")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs")
@click.option("--batch_size", default=2)
@click.option("--prompt", default=None)
def entry(model_id, data_path, class_path, seed, output_path, batch_size, prompt):
    main(model_id, data_path, class_path, seed, output_path, batch_size, prompt)


if __name__ == "__main__":
    entry()
