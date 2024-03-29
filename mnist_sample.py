import argparse
import os
import pickle
import subprocess
import zipfile

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from rich import print
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


class CustomizeDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_ret = self.data[index]
        labels_ret = self.labels[index]

        if self.transform is not None:
            data_ret = self.transform(self.data[index])

        data_ret = np.array(data_ret, dtype=np.float32)

        return data_ret, labels_ret


def parameter_setting():

    parser = argparse.ArgumentParser()

    parser.set_defaults(gpus='0', max_epochs=200)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--no_cuda', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epoch_number', type=int, default=1000)
    parser.add_argument('--lr', default=0.001)

    parser.add_argument('--data_root', default="../datasets/", type=str)
    parser.add_argument('--pytorch_data_path', default="../pytorch_data/")
    parser.add_argument('--result_folder', default="../output/", type=str)
    parser.add_argument('--lightning_log_folder',
                        default="../lightning_logs/", type=str)
    parser.add_argument('--download_require', default=True)
    parser.add_argument('--transform', default=None)

    args = parser.parse_args()

    import os
    os.makedirs(args.result_folder, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.pytorch_data_path, exist_ok=True)
    os.makedirs(args.lightning_log_folder, exist_ok=True)

    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    args.kwargs = {'num_workers': 8,
                   'pin_memory': True} if args.use_cuda else {}

    return args


class ClsMNIST(pl.LightningModule):

    def __init__(self, args):
        super(ClsMNIST, self).__init__()

        self.args = args

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, 10)

        self.do = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        """

        Args:
            x: The input image, (C,H, W)

        Returns:
            x: The output of neural network
        """

        x = x.view(-1, 28 * 28)

        # layer 1
        x = self.layer_1(x)

        x = torch.relu(x)
        x = self.do(x)

        # layer 2
        x = self.layer_2(x)

        x = torch.relu(x)
        x = self.do(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def prepare_data(self):

        # transform is used when applying dataloader
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])

        if not self.args.drive_download_require:
            mnist_train_data = MNIST(
                os.getcwd(), train=True, download=True, transform=transform)

            train_data_dict = {}
            train_data = mnist_train_data.data
            train_labels = mnist_train_data.targets

            train_data_list = [train_data[i].numpy()
                               for i in range(train_data.shape[0])]
            train_labels_list = [train_labels[i].numpy()
                                 for i in range(train_data.shape[0])]

            train_data_dict["data"] = train_data_list
            train_data_dict["labels"] = train_labels_list

            mnist_test_data = MNIST(
                os.getcwd(), train=False, download=True, transform=transform)

            test_data_dict = {}
            test_data = mnist_test_data.data
            test_labels = mnist_test_data.targets

            test_data_list = [test_data[i].numpy()
                              for i in range(test_data.shape[0])]
            test_labels_list = [test_labels[i].numpy()
                                for i in range(test_data.shape[0])]

            test_data_dict["data"] = test_data_list
            test_data_dict["labels"] = test_labels_list

            data_dict = {}

            data_dict["train"] = train_data_dict
            data_dict["test"] = test_data_dict

            with open(args.pytorch_data_path + args.dataset_name + ".pk", "wb") as pk_file:
                pickle.dump(data_dict, pk_file)

            train_dataset = mnist_train_data
            print("train_dataset: {}".format(len(train_dataset)))
            test_dataset = mnist_test_data

        else:
            file_name = os.path.join(
                args.data_root, args.dataset_name + ".zip")
            file_folder = os.path.join(args.data_root, args.dataset_name)

            # ****************************************************Downloading data ***************************************
            # file already exists
            if os.path.exists(file_name):
                # folder already exists
                if os.path.exists(file_folder):
                    print("Data is ready")
                # folder not exists; extract it
                else:
                    os.makedirs(file_folder, exist_ok=True)
                    with zipfile.ZipFile(file_name, "r") as zip_ref:
                        zip_ref.extractall(file_folder)
                    print("{}".format("Finish Extracting the Dataset ... "))
            # file not exists; download it
            else:
                print("{}".format("Downloading the Dataset ... "))
                subprocess.call(
                    ["sh", "./datasets_download_" + args.dataset_name + ".sh"])
                print("{}".format("Finish Downloading the Dataset ... "))
                os.makedirs(file_folder, exist_ok=True)
                with zipfile.ZipFile(file_name, 'r') as zip_ref:
                    zip_ref.extractall(file_folder)
                print("{}".format("Finish Extracting the Dataset ... "))

            with open(os.path.join(file_folder, args.dataset_name, args.dataset_name + ".pk"), "rb") as file:
                train_data = pickle.load(file)
                train_labels = pickle.load(file)
                test_data = pickle.load(file)
                test_labels = pickle.load(file)

            # print("train_data: {}, {}".format(train_data, train_data.max()))

            train_dict = {}
            train_dict["data"] = np.array(train_data)
            train_dict["labels"] = np.array(train_labels)

            test_dict = {}
            test_dict["data"] = np.array(test_data)
            test_dict["labels"] = np.array(test_labels)

            train_dataset = CustomizeDataset(
                train_dict, transform=args.transform)
            # print("{}".format(train_dataset))
            test_dataset = CustomizeDataset(
                test_dict, transform=args.transform)

            train_dataloader = DataLoader(
                train_dataset, batch_size=args.batch_size)
            test_dataloader = DataLoader(
                test_dataset, batch_size=args.batch_size)

        self.mnist_train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.mnist_test_loader = DataLoader(
            test_dataset, batch_size=self.args.batch_size)

    def train_dataloader(self):
        return self.mnist_train_loader

    def val_dataloader(self):
        return self.mnist_test_loader

    # def test_dataloader(self):
    #     return self.mnist_test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)

        train_prediction = logits.argmax(1)
        train_acc = train_prediction.eq(y).view(-1).float().mean()

        loss = self.cross_entropy_loss(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print("x shape: {}".format(x.size()))
        logits = self.forward(x)

        validation_prediction = logits.argmax(1)

        validation_loss = self.cross_entropy_loss(logits, y)
        validation_acc = validation_prediction.eq(y).view(-1).float().mean()

        self.log("val_loss", validation_loss, on_step=False, on_epoch=True)
        self.log("val_acc", validation_acc, on_step=False, on_epoch=True)


if __name__ == "__main__":
    cuda_index = "0"
    args = parameter_setting()
    # args.drive_download_require = False
    args.drive_download_require = True
    args.project_name = "mnist_mlp_pl_1"
    args.expr_index = 1
    args.layer_number = 3
    args.dataset_name = "MNIST"

    logger_name = "-".join(
        ["p-", args.project_name, "e-", str(args.expr_index), "l_n-", str(args.layer_number), "d_n-", args.dataset_name])
    logger = TensorBoardLogger(args.lightning_log_folder, name=logger_name)

    model = ClsMNIST(args)
    trainer = pl.Trainer(
        gpus=cuda_index, max_epochs=args.epoch_number, logger=logger)

    trainer.fit(model)
