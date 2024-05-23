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
import click
import os


def main(model_id, data_path, output_path, batch_size):
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

    if os.path.exists(output_path):
        outputs = [json.loads(line) for line in open(output_path)]
        data = data[len(outputs) :]

    with open(output_path, "a") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [Image.open(item["image"]) for item in batch]

            questions = [item["text"] for item in batch]

            if "llava-v1.6-mistral" in model_id:
                prompts = [
                    f"[INST] <image>\n{question}[/INST] Let's think step by step."
                    for question in questions
                ]
            elif "llava-v1.6-vicuna" in model_id:
                prompts = [
                    f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT: Let's think step by step."
                    for question in questions
                ]
            elif "blip" in model_id:
                prompts = [
                    f"Question: {question} Let's think step by step. Answer:"
                    for question in questions
                ]
            else:
                prompts = [
                    f"USER: <image>\n{question}\nASSISTANT: Let's think step by step."
                    for question in questions
                ]
            inputs = processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            ).to("cuda")
            output = model.generate(**inputs, max_new_tokens=1024)
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
            for item, text in zip(batch, generated_text):
                item["output"] = text
                if "mistral" in model_id:
                    item["pred"] = text.split("[/INST]")[-1].strip()
                elif "blip" in model_id:
                    item["pred"] = text.split("Answer:")[-1].strip()
                else:
                    item["pred"] = text.split("ASSISTANT:")[-1].strip()
                f.write(json.dumps(item) + "\n")
                f.flush()


@click.command()
@click.option("--model_id", default="ViT-B/32")
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--output_path", default="outputs.jsonl")
@click.option("--batch_size", default=8)
def entry(model_id, data_path, output_path, batch_size):
    main(model_id, data_path, output_path, batch_size)


if __name__ == "__main__":
    entry()
