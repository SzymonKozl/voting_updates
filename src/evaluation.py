from typing import Callable, List, Type, Optional, Any, Literal

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import torch
import torchvision
from sklearn.metrics import classification_report
import abc
from tqdm import tqdm

from functorch import make_functional, vmap, grad

from src.mes import calculate_mes, binary_decsisions_feasible_updater, config_numpy


def _get_valuations_vec(valuations_raw: torch.Tensor):
    msk = valuations_raw < 0
    N = valuations_raw.size()[0]
    M = valuations_raw.size()[1]
    res = torch.zeros(size=(N, M * 2))
    res[:, :M] += valuations_raw * torch.logical_not(msk)
    res[:, M:] += valuations_raw * -1. * msk
    return res


class BatchStrategicTrainer(abc.ABC):
    def __init__(self, model: nn.Module, **kwargs: Any):
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    @abc.abstractmethod
    def train_batch(self, images: torch.Tensor, labels: torch.Tensor):
        pass


class BaseBatchStrategicTrainer(BatchStrategicTrainer):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.optim = torch.optim.Adam(self.model.parameters())

    def train_batch(self, images: torch.Tensor, labels: torch.Tensor):
        self.optim.zero_grad()
        outputs = self.model(images)
        _, predicted = torch.max(outputs, 1)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optim.step()


class MESBatchStrategicTrainer(BatchStrategicTrainer):
    def __init__(self, model: nn.Module, mode: Literal["single", "layerwise"] = "layerwise"):
        super().__init__(model)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.model_functional, self.model_params = make_functional(model)
        self.update_rate = 1e-4
        self.mode = mode

    def _apply_update_on_layer(self, layer_ix: int, update_mask: torch.Tensor):
        self.model_params[layer_ix].data += self.update_rate * update_mask

    def train_batch(self, images: torch.Tensor, labels: torch.Tensor):
        def _loss_fn(params, x, y):
            output = self.model_functional(params, x)
            return self.loss(output, y)

        grad_func = grad(_loss_fn)
        batched_grad_func = vmap(grad_func, in_dims=(None, 0, 0))
        grads = batched_grad_func(self.model_params, images, labels)

        if self.mode == "single":
            pass
            # concatenate gradients into vector of shape batch_size X param_no

            # calc MES on this shit

            # decompose concatenated MES outcome and apply updates
        else:
            for i, grads_layer in enumerate(grads):
                grads_layer = grads_layer.detach()
                shape = grads_layer.size()
                grads_layer = torch.flatten(grads_layer, start_dim=1)
                valuations = _get_valuations_vec(grads_layer)
                chosen = calculate_mes(valuations.T.numpy(), feasible_updater=binary_decsisions_feasible_updater, safe=False, verbose=True)
                M = valuations.size()[1] // 2
                update = chosen
                update[M:] *= -1.
                update *= self.update_rate
                update = update[:M] + update[M:]
                update = torch.reshape(torch.Tensor(update), shape[1:])
                with torch.no_grad():
                    self.model_params[i].data += update


model_getter = Callable[[], nn.Module]


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


def evaluate_training_strategy(
        get_model: model_getter,
        trainer_type: Type[BatchStrategicTrainer],
        ds_train: DataLoader,
        ds_test: DataLoader,
        repeats: int = 10,
        epochs: int = 15,
        trainer_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    if trainer_kwargs is None:
        trainer_kwargs = {}
    reports = []
    for _ in range(repeats):
        net = get_model()
        net.train()
        trainer = trainer_type(net, **trainer_kwargs)
        for _ in tqdm(range(epochs)):
            for images, labels in tqdm(ds_train):
                images, labels = images.to(device), labels.to(device)
                trainer.train_batch(images, labels)
        reports.append(get_report(net, ds_test))
    return pd.DataFrame(avg_report(reports))


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.dense = nn.Linear(32 * 22 * 22, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=-3)
        x = torch.softmax(self.dense(x), dim=-1)
        return x


if __name__ == '__main__':
    config_numpy()
    def _get_model():
        return SmallModel()


    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    evaluate_training_strategy(_get_model, MESBatchStrategicTrainer, train_loader, test_loader, 5)
