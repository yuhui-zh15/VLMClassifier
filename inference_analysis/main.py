import json
import random
import os

import click
import torch
from PIL import Image
from tqdm import tqdm, trange
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
)
from typing import List


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_image(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    image_file: str,
    classes: List[str],
    prompt: str,
):
    # print(processor("This is a good day"))
    # print(processor("This is a good"))
    # print(processor("day"))
    # {'input_ids': tensor([[   1,  910,  338,  263, 1781, 2462]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]]), 'pixel_values': None}
    # {'input_ids': tensor([[   1,  910,  338,  263, 1781]]), 'attention_mask': tensor([[1, 1, 1, 1, 1]]), 'pixel_values': None}
    # {'input_ids': tensor([[   1, 2462]]), 'attention_mask': tensor([[1, 1]]), 'pixel_values': None}

    with torch.no_grad():
        image = Image.open(image_file)
        first_inputs = processor(text=prompt, images=image, return_tensors="pt")
        for key in first_inputs:
            first_inputs[key] = first_inputs[key].to(device)
        first_outputs = model(**first_inputs, use_cache=True)

        probs = []
        for classname in tqdm(classes):
            inputs = processor(text=classname)
            input_ids = inputs.input_ids[0, 1:].numpy().tolist()  # batch size 1

            outputs = first_outputs
            past_key_values = outputs.past_key_values

            probs.append([])
            for input_id in input_ids:
                logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                probs[-1].append(log_probs[input_id].item())

                outputs = model(
                    input_ids=torch.tensor([[input_id]], device=device),
                    pixel_values=first_inputs["pixel_values"],
                    attention_mask=first_inputs["attention_mask"],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        return probs


def process_image_raw(
    model: LlavaForConditionalGeneration,
    processor: LlavaProcessor,
    image_file: str,
    classes: List[str],
    prompt: str,
):
    with torch.no_grad():
        image = Image.open(image_file)
        probs = []
        for classname in tqdm(classes):
            inputs = processor(
                text=f"{prompt} {classname}", images=image, return_tensors="pt"
            )
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            # print(processor(text=classname))
            class_ids = torch.tensor(processor(text=classname)["input_ids"][1:]).to(
                device
            )
            selected_logits = logits[0, -len(class_ids) - 1 : -1, :]
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)
            log_prob = (
                torch.gather(log_probs, 1, class_ids.unsqueeze(1))
                .squeeze(1)
                .cpu()
                .float()
                .numpy()
                .tolist()
            )
            probs.append(log_prob)

        return probs


@click.command()
@click.option("--model_id", default="ViT-B/32")
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--class_path", default="../data/imagenet_classes.json")
@click.option("--split", default="valid")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs.jsonl")
def main(model_id, data_path, class_path, split, seed, output_path):
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
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))
    data = data[:1000]

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    if os.path.exists(output_path):
        print("Resuming from existing output file")
        outputs = [json.loads(line) for line in open(output_path)]
        data = data[len(outputs) :]

    with open(output_path, "a") as f:
        for idx in trange(len(data)):
            if "blip" in model_id:
                probs = process_image_raw(
                    model,
                    processor,
                    data[idx]["image"],
                    classes,
                    "Question: What type of object is in this photo? Answer: A",
                )
            else:
                probs = process_image(
                    model,
                    processor,
                    data[idx]["image"],
                    classes,
                    "USER: <image>\nWhat type of object is in this photo?\nASSISTANT: The photo features a",
                )

            data[idx]["label_index"] = classes.index(data[idx]["label"])
            data[idx]["probs"] = probs

            f.write(json.dumps(data[idx]) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
