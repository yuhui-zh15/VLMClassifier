import numpy as np
import torch
import json
import math
import random
import click
from tqdm import trange
from matplotlib import pyplot as plt


class LinearProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LinearProbing, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbing(torch.nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(MLPProbing, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim, feature_dim)
        self.fc2 = torch.nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@click.command()
@click.option("--dataset", default="flowers")
@click.option("--model_name", default="llava7b")
@click.option("--probe", default="linear")
@click.option("--split", default="test")
@click.option("--feature_type", default="last")
@click.option("--n_epochs", default=500)
def main(dataset, model_name, probe, split, feature_type, n_epochs):
    data = [json.loads(line) for line in open(f"../data/{dataset}.jsonl")]
    random.seed(1234)
    random.shuffle(data)
    classes = json.load(open(f"../data/{dataset}_classes.json"))

    labels = []
    for item in data:
        item["label_index"] = classes.index(item["label"])
        labels.append(item["label_index"])
    labels = torch.tensor(labels)

    n_shards = math.ceil(len(data) // 1024) + 2
    print(len(data), n_shards)

    features = []
    for i in range(n_shards):
        feature = torch.load(f"outputs/{dataset}_{model_name}_{i}.pt")
        features.append(feature)

    features = torch.cat(features, dim=0).float()
    print(features.shape)

    train_idxs = [i for i in range(len(data)) if data[i]["split"] == "train"]
    test_idxs = [i for i in range(len(data)) if data[i]["split"] == split]

    feature_idx = None
    if feature_type == "last":
        feature_idx = 0
    elif feature_type == "avg":
        feature_idx = 1
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    train_features = features[train_idxs, feature_idx]
    test_features = features[test_idxs, feature_idx]
    train_labels = labels[train_idxs]
    test_labels = labels[test_idxs]

    print(
        train_features.shape, test_features.shape, train_labels.shape, test_labels.shape
    )

    if probe == "linear":
        model = LinearProbing(len(train_features[0]), len(classes)).cuda()
    elif probe == "mlp":
        model = MLPProbing(len(train_features[0]), len(classes)).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    bsz = 512

    accs = []
    for epoch in trange(n_epochs):
        for i in range(0, len(train_features), bsz):
            optimizer.zero_grad()
            output = model(train_features[i : i + bsz].cuda())
            loss = criterion(output, train_labels[i : i + bsz].cuda())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            eval_bsz = 512
            preds = []
            for i in range(0, len(train_features), eval_bsz):
                output = model(train_features[i : i + eval_bsz].cuda())
                pred = output.argmax(dim=1).cpu()
                preds.append(pred)
            preds = torch.cat(preds)
            train_acc = (preds == train_labels).float().mean().item()

            preds = []
            for i in range(0, len(test_features), eval_bsz):
                output = model(test_features[i : i + eval_bsz].cuda())
                pred = output.argmax(dim=1).cpu()
                preds.append(pred)
            preds = torch.cat(preds)
            test_acc = (preds == test_labels).float().mean().item()

            accs.append((train_acc, test_acc))

    plt.plot([train_acc for train_acc, _ in accs], label="train")
    plt.plot([test_acc for _, test_acc in accs], label="test")
    plt.legend()

    output_prefix = (
        f"probe_outputs/{dataset}_{model_name}_{probe}_{split}_{feature_type}"
    )
    plt.savefig(f"{output_prefix}.png")

    print(max([test_acc for _, test_acc in accs]))
    torch.save([accs, model.state_dict()], f"{output_prefix}.pt")


if __name__ == "__main__":
    main()
