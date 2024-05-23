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
import torch.nn.functional as F
from PIL import Image
import json
from tqdm import trange
import random
import click
import clip
from torchvision import transforms
from torchvision.models import vit_b_16
import os

# Comment out only if using the EVA-CLIP model
########################################################################################
# import sys
# sys.path.append("./VlmClassifier/EVA/EVA-CLIP/rei")
# from eva_clip import create_model_and_transforms, get_tokenizer
########################################################################################


def main(
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    chain_of_thought,
    batch_size,
    fixed_order,
    init_prompt=None,
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
    data = [item for item in data if item["split"] == split]

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

    with open(output_path, "a") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [Image.open(item["image"]) for item in batch]

            if including_label:
                choices = []
                for item in batch:
                    if fixed_order:
                        assert n_labels == len(classes)
                        choices.append(classes)
                    else:
                        label = item["label"]
                        other_choices = random.sample(
                            sorted(list(set(classes) - set([label]))), n_labels - 1
                        )
                        shuffled_choices = [label] + other_choices

                        random.shuffle(shuffled_choices)
                        choices.append(shuffled_choices)
                questions = [
                    f"{init_prompt} Choose one from \"{', '.join(choice)}\"."
                    for choice in choices
                ]
            else:
                questions = [init_prompt for _ in batch]

            if "llava-v1.6-mistral" in model_id:
                assert not chain_of_thought
                prompts = [
                    f"[INST] <image>\n{question}[/INST]" for question in questions
                ]
            elif "llava-v1.6-vicuna" in model_id:
                assert not chain_of_thought
                prompts = [
                    f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
                    for question in questions
                ]
            elif "blip" in model_id:
                if not chain_of_thought:
                    prompts = [
                        f"Question: {question} Answer:" for question in questions
                    ]
                else:
                    prompts = [
                        f"Question: {question} Let's think step by step. Answer:"
                        for question in questions
                    ]
            else:
                if not chain_of_thought:
                    prompts = [
                        f"USER: <image>\n{question}\nASSISTANT:"
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
            output = model.generate(
                **inputs, max_new_tokens=64 if not chain_of_thought else 512
            )
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


def main_clip(
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    batch_size,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "eva" in model_id.lower():
        pretrained = "eva_clip"
        model, _, preprocess = create_model_and_transforms(
            model_id, pretrained, force_custom_clip=True
        )
        tokenizer = get_tokenizer(model_id)
        model = model.to(device)
    else:
        model, preprocess = clip.load(model_id, device=device)
        tokenizer = clip.tokenize

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    with open(output_path, "w") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [
                preprocess(Image.open(item["image"]).convert("RGB")).unsqueeze(0)
                for item in batch
            ]
            image_tensor = torch.cat(images).to(device)

            text_descriptions = [f"A photo of a {cls}" for cls in classes]
            text_tokens = tokenizer(text_descriptions).to(device)

            if including_label:
                choices = []
                for item in batch:
                    label = item["label"]
                    other_choices = random.sample(
                        sorted(list(set(classes) - set([label]))), n_labels - 1
                    )
                    shuffled_choices = [label] + other_choices

                    random.shuffle(shuffled_choices)
                    indices = [classes.index(choice) for choice in shuffled_choices]
                    choices.append(indices)
            else:
                choices = None

            with torch.no_grad():
                image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
                text_features = F.normalize(model.encode_text(text_tokens), dim=-1)

                similarities = image_features @ text_features.T

            for idx in range(len(batch)):
                item = batch[idx]
                if including_label:
                    choice = choices[idx]
                    pred = similarities[idx][choice].argmax()
                    predicted_class = classes[choice[pred.item()]]
                else:
                    predicted_class = classes[similarities[idx].argmax().item()]

                item["choices"] = (
                    [classes[class_idx] for class_idx in choices[idx]]
                    if including_label
                    else None
                )
                item["pred"] = predicted_class
                f.write(json.dumps(item) + "\n")
                f.flush()


def main_supervised(
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    batch_size,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_b_16(pretrained=True).to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data = [json.loads(line) for line in open(data_path)]
    data = [item for item in data if item["split"] == split]

    random.seed(seed)
    random.shuffle(data)
    classes = json.load(open(class_path))

    print(f"{len(data)=}")
    print(f"{len(set(classes))=}")

    with open(output_path, "w") as f:
        for i in trange(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            images = [
                preprocess(Image.open(item["image"]).convert("RGB")).unsqueeze(0)
                for item in batch
            ]
            image_tensor = torch.cat(images).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                _, preds = torch.max(outputs, 1)

            for idx in range(len(batch)):
                item = batch[idx]
                predicted_class = classes[preds[idx].item()]

                if including_label:
                    label = item["label"]
                    other_choices = random.sample(
                        sorted(list(set(classes) - set([label]))), n_labels - 1
                    )
                    choices = [label] + other_choices
                    random.shuffle(choices)
                    item["choices"] = choices
                else:
                    item["choices"] = None

                item["pred"] = predicted_class
                f.write(json.dumps(item) + "\n")
                f.flush()


@click.command()
@click.option("--method", default="vlm")
@click.option("--model_id", default="ViT-B/32")
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--class_path", default="../data/imagenet_classes.json")
@click.option("--split", default="valid")
@click.option("--seed", default=1234)
@click.option("--output_path", default="outputs.jsonl")
@click.option("--including_label", default=False)
@click.option("--n_labels", default=1000)
@click.option("--chain_of_thought", default=False)
@click.option("--batch_size", default=8)
@click.option("--fixed_order", default=False)
@click.option("--prompt", default=None)
def entry(
    method,
    model_id,
    data_path,
    class_path,
    split,
    seed,
    output_path,
    including_label,
    n_labels,
    chain_of_thought,
    batch_size,
    fixed_order,
    prompt,
):
    if method == "vlm":
        main(
            model_id,
            data_path,
            class_path,
            split,
            seed,
            output_path,
            including_label,
            n_labels,
            chain_of_thought,
            batch_size,
            fixed_order,
            prompt,
        )
    elif method == "clip":
        main_clip(
            model_id,
            data_path,
            class_path,
            split,
            seed,
            output_path,
            including_label,
            n_labels,
            batch_size,
        )
    elif method == "supervised":
        main_supervised(
            model_id,
            data_path,
            class_path,
            split,
            seed,
            output_path,
            including_label,
            n_labels,
            batch_size,
        )
    else:
        raise ValueError(f"Invalid method: {method}")


if __name__ == "__main__":
    entry()
