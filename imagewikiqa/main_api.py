from PIL import Image
import json
from tqdm import tqdm
import random
import click
import os
import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def load_data(data_path):
    data = [json.loads(line) for line in open(data_path)]
    return data


def process_with_gemini(item, processed_images):
    if item["image"] in processed_images:
        return None

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name="gemini-pro-vision")

    image = Image.open(item["image"])
    question = create_question(item)

    output = None
    for i in range(3):
        try:
            response = model.generate_content([question, image])
            output = response.text
        except Exception as e:
            print(e, item, f"{i} times retrying...")
            time.sleep(60)
            continue
    if output is None:
        print("No response", item)
        return None

    item["question"] = question
    item["output"] = output
    return item


def process_with_claude(item, processed_images):
    if item["image"] in processed_images:
        return None

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    with open(item["image"], "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode("utf-8")

    question = create_question(item)

    output = None
    for i in range(3):
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_string,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
            output = response.content[0].text
        except Exception as e:
            print(e, item, f"{i} times retrying...")
            time.sleep(60)
            continue
    if output is None:
        print("No response", item)
        return None

    item["question"] = question
    item["output"] = output
    return item


def process_with_gpt4v(item, processed_images):
    if item["image"] in processed_images:
        return None

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    img_path = item["image"]
    base64_image = encode_image(img_path)

    question = create_question(item)

    output = None
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
            )
            output = response.choices[0].message.content
        except Exception as e:
            print(e, item, f"{i} times retrying...")
            time.sleep(60)
            continue
    if output is None:
        print("No response", item)
        return None

    item["question"] = question
    item["output"] = output
    return item


def create_question(item):
    question = f"{item['text']} Let's think step by step."
    return question


@click.command()
@click.option("--data_path", default="../data/imagenet.jsonl")
@click.option("--output_path", default="./outputs/outputs.jsonl")
@click.option(
    "--api", type=click.Choice(["gemini", "claude", "gpt4v"]), default="gemini"
)
@click.option("--threads", default=16)
@click.option("--data_limit", default=10000)
def main(data_path, output_path, api, threads, data_limit):
    data = load_data(data_path)
    data = data[:data_limit]

    if os.path.exists(output_path):
        outputs = [json.loads(line) for line in open(output_path)]
        processed_images = set([item["image"] for item in outputs])
    else:
        processed_images = set()

    with ThreadPoolExecutor(max_workers=threads) as executor, open(
        output_path, "a"
    ) as f:
        futures = []
        for item in data:
            if api == "gemini":
                futures.append(
                    executor.submit(process_with_gemini, item, processed_images)
                )
            elif api == "claude":
                futures.append(
                    executor.submit(process_with_claude, item, processed_images)
                )
            elif api == "gpt4v":
                futures.append(
                    executor.submit(process_with_gpt4v, item, processed_images)
                )

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
