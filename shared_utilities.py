import csv
import sys

import PIL
import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Dataset


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, cos_t_max):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.cos_t_max = cos_t_max

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # return [optimizer], [sch]
        # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.cos_t_max)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",  # default
                "frequency": 1,  # default
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        feature = batch[0]
        logits = self(feature)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels
        # return self.model( batch  )


class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_path="./", batch_size=64, num_workers=2, train_transform=None, test_transform=None):
        super().__init__()
        self.train_df = self.predict_df = None  # load these in prepare_data
        self.train = self.valid = self.test = self.predict = None  # create these in setup
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        # Download, tokenize, etc.
        # load the data into tensors via Pandas
        self.train_df = pd.read_csv(f'{self.data_path}train.csv')
        self.predict_df = pd.read_csv(f'{self.data_path}test.csv')

    def setup(self, stage=None):
        # split, transform, etc.
        # create train, test and valid sets by splitting the training data
        train_targets = self.train_df.label.values
        train_features = self.train_df.drop('label', axis=1).values / 255  # normalized
        train_features = train_features.reshape(-1, 1, 28, 28)
        train_targets_tensor = torch.from_numpy(train_targets).type(torch.long)
        train_features_tensor = torch.from_numpy(train_features).type(torch.float32)

        # train = torch.utils.data.TensorDataset(train_features_tensor, train_targets_tensor)
        train = MyDataset(train_features_tensor, train_targets_tensor, self.train_transform)
        self.train, self.test, self.valid = random_split(train, lengths=[0.80, 0.15, 0.05],
                                                         generator=torch.Generator())

        # create predict dataset from the test.csv file
        predict_features = self.predict_df.values / 255
        predict_features = predict_features.reshape(-1, 1, 28, 28)
        predict_tensor = torch.from_numpy(predict_features).type(torch.float32)
        # self.predict = torch.utils.data.TensorDataset(predict_tensor)
        self.predict = MyDataset(predict_tensor, transform=self.test_transform)
        # print( f'predict_tensor={predict_tensor}')
        # print( f'self.predict (TensorDataSet)={self.predict}')

    def predict_dataloader(self):
        predict_loader = DataLoader(
            dataset=self.predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        return predict_loader

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
        return test_loader


class MyDataset(Dataset):
    def __init__(self, image_tensor, labels=None, transform=None):
        self.transform = transform
        self.labels = labels
        self.image_tensor = image_tensor

    def __getitem__(self, index):
        # Load image as NumPy array
        image_array = self.image_tensor[index].squeeze().numpy()  # Assuming single-channel
        image = PIL.Image.fromarray(image_array)
        if self.transform:
            image = self.transform(image)

        label = '0'
        if self.labels is not None:
            label = self.labels[index]

        return image, label

    def __len__(self):
        return len(self.image_tensor)


def plot_loss_and_acc(
        log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.7, 1.0), save_loss=None, save_acc=None
):
    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)


class PyTorchCNN(torch.nn.Module):
    def __init__(self, num_classes, dropout_rate=0.25):
        super().__init__()

        self.cnn_layers = torch.nn.Sequential(

            torch.nn.Conv2d(1, 3, kernel_size=5),
            torch.nn.BatchNorm2d(3),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout2d(p=dropout_rate),

            torch.nn.Conv2d(3, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout2d(p=dropout_rate),

            torch.nn.Conv2d(16, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            # torch.nn.ReLU(),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.fc_layers = torch.nn.Sequential(
            # hidden layer
            torch.nn.Linear(32, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(20, 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_classes)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc_layers(x)
        return logits


class MyProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


def save_predictions_to_csv(predictions, filename="predictions.csv"):
    """
    Saves predictions as a CSV file.

    Args:
        predictions: A list of PyTorch tensors representing batched predictions.
        filename (str, optional): The filename for the CSV file. Defaults to "predictions.csv".
    """
    # Combine all predictions into a single list
    all_predictions = []
    for batch_predictions in predictions:
        all_predictions.extend(batch_predictions.cpu().tolist())  # Convert to list and move to CPU

    # Generate image IDs starting from 1 (assuming consecutive numbering)
    image_ids = range(1, len(all_predictions) + 1)

    # Create a list of tuples for writing to CSV
    data = list(zip(image_ids, all_predictions))

    # Write data to CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "Label"])  # Write header row
        writer.writerows(data)

    print(f"Predictions saved to CSV file: {filename}")
