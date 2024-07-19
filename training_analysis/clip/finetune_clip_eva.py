import math
import os
import torch

# import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import json
from torchvision import transforms
import argparse
import pdb
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import time
import sys

sys.path.append("./VlmClassifier/EVA/EVA-CLIP/rei")
from eva_clip import create_model_and_transforms, get_tokenizer


class ParsedDataset(Dataset):
    def __init__(
        self,
        data_path,
        transform=None,
        split="test",
        preprocess=None,
        imagenet=False,
        classes=None,
    ):
        data = [json.loads(line) for line in open(data_path)]
        self.data = [item for item in data if item["split"] == split]
        self.transform = transform
        self.split = split
        self.preprocess = preprocess
        self.imagenet = imagenet
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(Image.open(item["image"]).convert("RGB"))
        if self.imagenet:
            return image, item["label"], item["idx"], item["label_idx"]
        else:
            item["label_idx"] = self.classes.index(item["label"])
            return image, item["label"], -1, item["label_idx"]


def main_finetune_linear_only(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = json.load(open(args.class_path))

    print("Preparing Datasets")
    is_imagenet = "imagenet" in args.data_path
    data = [json.loads(line) for line in open(args.data_path)]
    train_data = [item for item in data if item["split"] == "train"]
    test_split = "valid" if is_imagenet else "test"
    test_data = [item for item in data if item["split"] == test_split]

    train_labels = []
    for item in train_data:
        if not is_imagenet:
            item["label_idx"] = classes.index(item["label"])
        train_labels.append(item["label_idx"])
    train_labels = torch.tensor(train_labels)

    test_labels = []
    for item in test_data:
        if not is_imagenet:
            item["label_idx"] = classes.index(item["label"])
        test_labels.append(item["label_idx"])
    test_labels = torch.tensor(test_labels)

    n_shards = math.ceil(len(train_data) // 1024) + 1
    print(len(train_data), n_shards)
    # print("Loading Features")
    if args.is_eva:
        feature_path = f"train_analysis/dev_output/{args.dataset}_feature_eva"
    else:
        feature_path = f"train_analysis/dev_output/{args.dataset}_feature"
    print("Loading features from ", feature_path)
    train_features = []
    for i in tqdm(range(n_shards)):
        train_feature = torch.load(os.path.join(feature_path, "train", f"{i}.pt"))
        train_features.append(train_feature)
    train_features = torch.cat(train_features, dim=0).float()
    train_len = len(train_features)
    print(train_features.shape)

    n_shards = math.ceil(len(test_data) // 1024) + 1
    print(len(test_data), n_shards)
    test_features = []
    for i in range(n_shards):
        test_feature = torch.load(os.path.join(feature_path, "test", f"{i}.pt"))
        test_features.append(test_feature)
    test_features = torch.cat(test_features, dim=0).float()
    test_len = len(test_features)
    print(test_features.shape)

    bsz = 512
    print("Only Finetuning Linear Head")
    num_classes = len(classes)
    if args.is_eva:
        feat_dim = 1024
    else:
        feat_dim = 768
    classifier = LinearProbing(feat_dim, num_classes).to(device).float()
    for name, p in classifier.named_parameters():
        if p.requires_grad:
            print(name)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    classifier.eval()
    eval_bsz = 256
    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, test_len, eval_bsz)):
            outputs = classifier(
                F.normalize(test_features[i : i + eval_bsz].cuda(), dim=-1)
            )
            pred_label = outputs.argmax(dim=-1).cpu().numpy()
            preds.extend(pred_label)
    preds = np.array(preds)
    gt_labels = np.array(test_labels)
    acc = (preds == gt_labels).mean()
    acc_str = f"Before Training, Acc on test: {acc * 100:.2f}%"
    print(acc_str)
    output_log = os.path.join(args.output_path, f"linear_only_{args.dataset}_log.txt")
    with open(output_log, "w") as f:
        for epoch in range(args.epochs):
            classifier.train()
            shuffled_idxs = list(range(train_len))
            random.shuffle(shuffled_idxs)
            shuffled_train_feats = train_features[shuffled_idxs]
            shuffled_train_labels = train_labels[shuffled_idxs]
            log_flag = 1
            for i in range(0, train_len, bsz):
                log_flag += 1
                outputs = classifier(
                    F.normalize(shuffled_train_feats[i : i + bsz].cuda(), dim=-1)
                )
                loss = criterion(outputs, shuffled_train_labels[i : i + bsz].cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if log_flag % 1000 == 0:
                    loss_str = f"Step {log_flag} in Epoch {epoch}, Loss: {loss.item()}"
                    print(loss_str)
                    f.write(loss_str + "\n")
            classifier.eval()
            eval_bsz = 256
            preds = []
            with torch.no_grad():
                for i in tqdm(range(0, test_len, eval_bsz)):
                    outputs = classifier(
                        F.normalize(test_features[i : i + eval_bsz].cuda(), dim=-1)
                    )
                    pred_label = outputs.argmax(dim=-1).cpu().numpy()
                    preds.extend(pred_label)
            preds = np.array(preds)
            gt_labels = np.array(test_labels)
            acc = (preds == gt_labels).mean()
            acc_str = f"Epoch {epoch}, Acc on test: {acc * 100:.2f}%"
            print(acc_str)
            f.write(acc_str + "\n")
    model_save_path = os.path.join(args.output_path, f"{args.dataset}_linear.pt")
    torch.save(classifier.state_dict(), model_save_path)
    print("Model saved to ", model_save_path)


class LinearProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbing, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes).to(torch.float16)

    def forward(self, x):
        return self.fc(x)


def extract_features(args):
    print("Loading CLIP model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "eva" in args.model_id.lower():
        pretrained = "eva_clip"
        model, _, preprocess = create_model_and_transforms(
            args.model_id, pretrained, force_custom_clip=True
        )
        tokenizer = get_tokenizer(args.model_id)
        model = model.to(device)
    else:
        model, preprocess = clip.load(args.model_id, device=device)
        tokenizer = clip.tokenize
    classes = json.load(open(args.class_path))
    print("Preparing Datasets")
    is_imagenet = "imagenet" in args.data_path

    train_dataset = ParsedDataset(
        args.data_path,
        split="train",
        transform=preprocess,
        imagenet=is_imagenet,
        classes=classes,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=False, num_workers=8
    )

    test_split = "valid" if is_imagenet else "test"
    test_dataset = ParsedDataset(
        args.data_path,
        split=test_split,
        transform=preprocess,
        imagenet=is_imagenet,
        classes=classes,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=8
    )
    print("Extract Features Only")
    if args.is_eva:
        save_dir_train = f"train_analysis/dev_output/{args.dataset}_feature_eva/train"
        save_dir_test = f"train_analysis/dev_output/{args.dataset}_feature_eva/test"

    else:
        save_dir_train = f"train_analysis/dev_output/{args.dataset}_feature/train"
        save_dir_test = f"train_analysis/dev_output/{args.dataset}_feature/test"
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            images, labels, uq_idx, label_idx = batch
            images = images.to(device)
            image_features = model.encode_image(images)
            save_path = os.path.join(save_dir_train, f"{batch_idx}.pt")
            torch.save(image_features, save_path)

        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, labels, uq_idx, label_idx = batch
            images = images.to(device)
            image_features = model.encode_image(images)
            save_path = os.path.join(save_dir_test, f"{batch_idx}.pt")
            torch.save(image_features, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--model_id", type=str, default="ViT-L/14@336px")
    parser.add_argument("--data_path", type=str, default="data/imagenet.jsonl")
    parser.add_argument("--class_path", type=str, default="data/imagenet_classes.json")
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--output_path", type=str, default="train_analysis/dev_output")
    parser.add_argument("--seed", default=0, type=int, help="random seed to use")

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # device = torch.device('cuda:0')
    # args.device = device
    args.data_path = f"data/{args.dataset}.jsonl"
    args.class_path = f"data/{args.dataset}_classes.json"
    args.is_eva = "eva" in args.model_id.lower()
    if args.is_eva:
        args.output_path = os.path.join(args.output_path, f"{args.dataset}_eva")
    else:
        args.output_path = os.path.join(args.output_path, args.dataset)
    extract_features(args)
    main_finetune_linear_only(args)
