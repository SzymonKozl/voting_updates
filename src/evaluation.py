from typing import Callable, List

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import classification_report


model_getter = Callable[[], nn.Module]
update_strategy = Callable[[nn.Module], None]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_report(net: nn.Module, test_loader: DataLoader) -> dict:
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return classification_report(all_labels, all_predictions, output_dict=True)


def avg_report(reports: List[dict]) -> dict:
    res = {}
    for entry in zip(*[r.items() for r in reports]):
        if isinstance(entry[0][1], float):
            res[entry[0][0]] = sum(t[1] for t in entry) / len(reports)
        else:
            res[entry[0][0]] = avg_report([t[1] for t in entry])
    return res


# "grand evaluation framework"
def evaluate_training_strategy(
        get_model: model_getter,
        update_weights: update_strategy,
        ds_train: DataLoader,
        ds_test: DataLoader,
        repeats: int = 10,
        epochs: int = 15
) -> pd.DataFrame:
    reports = []
    for _ in range(repeats):
        net = get_model()
        net.train()
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for images, labels in ds_train:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)
                loss.backward()
                update_weights(net)
        reports.append(get_report(net, ds_test))
    return pd.DataFrame(avg_report(reports))
