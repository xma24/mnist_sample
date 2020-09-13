import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import MNIST
import os
from torchvision import datasets, transforms

import numpy as np
import pickle


class CustomizeDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        # self.data = torch.FloatTensor(data.values.astype('float'))

        self.transform = transform

        self.data = data_dict["data"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # if self.transform is not None:
        #     self.data = [self.transform(i) for i in self.data]

        data_ret = self.data[index]
        # print("data_ret: {}, {}".format(data_ret, data_ret.shape))
        labels_ret = self.labels[index]

        if self.transform is not None:
            data_ret = self.transform(self.data[index])

        data_ret = np.array(data_ret, dtype=np.float32)

        return data_ret, labels_ret


import argparse
import os


def parameter_setting(cuda_index):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='usps')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--no_cuda', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epoch_number', type=int, default=1000)
    parser.add_argument('--lr', default=0.001)

    parser.add_argument('--using_torch_dataset', default=False)
    parser.add_argument('--data_path', default="./data/")


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    os.makedirs(args.data_path, exist_ok=True)

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda:" + str(cuda_index) if use_cuda else "cpu")
    print("using device: {} ".format(args.device))

    args.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    return args


class LightningMNISTClassifier(pl.LightningModule):

    def __init__(self, args):
        super(LightningMNISTClassifier, self).__init__()

        self.args = args

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 64)
        self.layer_3 = torch.nn.Linear(64, 10)

        self.do = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # batch_size, channels, width, height = x.size()
        # print("x shape: {}".format(x.size()))

        x0 = x[1]
        for i in range(x0.shape[0]):
            print("x0[i]: {}".format(x0[i]))
        #
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(-1, 28 * 28)
        print("x shape: {}, {}".format(x, x.max()))


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

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

        # transformations = transforms.Compose([transforms.Scale(32),transforms.ToTensor()])


        if self.args.using_torch_dataset:
            mnist_train_data = MNIST(os.getcwd(), train=True, download=True, transform=transform)

            train_data_dict = {}
            train_data = mnist_train_data.data
            train_labels = mnist_train_data.targets
            # print("train_data: {}, {}".format(train_data.max(), train_data.shape))
            # print("train_labels: {}, {}".format(train_labels, train_labels.shape))

            train_data_list = [ train_data[i].numpy() for i in range(train_data.shape[0])]
            train_labels_list = [ train_labels[i].numpy() for i in range(train_data.shape[0])]

            # train_data_list = np.array(train_data.numpy(), dtype=np.float32).tolist()
            # train_labels_list = np.array(train_labels.numpy(), dtype=np.float32).tolist()

            print("train_data_list[6]: {}, {}".format(train_data_list[6], train_data_list[6].shape))



            train_data_dict["data"] = train_data_list
            print("train_data_dict[data]: {}".format(len(train_data_dict["data"])))
            train_data_dict["labels"] = train_labels_list


            mnist_test_data = MNIST(os.getcwd(), train=False, download=True, transform=transform)

            test_data_dict = {}
            test_data = mnist_test_data.data
            test_labels = mnist_test_data.targets
            # print("test_data: {}, {}".format(test_data.max(), test_data.shape))
            # print("test_labels: {}, {}".format(test_labels, test_labels.shape))


            test_data_list = [ test_data[i].numpy() for i in range(test_data.shape[0])]
            test_labels_list = [ test_labels[i].numpy() for i in range(test_data.shape[0])]

            # test_data_list = np.array(test_data.numpy(), dtype=np.float32).tolist()
            # test_labels_list = np.array(test_labels.numpy(), dtype=np.float32).tolist()

            test_data_dict["data"] = test_data_list
            test_data_dict["labels"] = test_labels_list

            data_dict = {}

            data_dict["train"] =  train_data_dict
            data_dict["test"] = test_data_dict

            with open(args.data_path + args.name + ".pk", "wb") as pk_file:
                pickle.dump(data_dict, pk_file)

            train_dataset = mnist_train_data
            print("train_dataset: {}".format(len(train_dataset)))
            test_dataset = mnist_test_data

        else:
            with open(args.data_path + args.name + ".pk", "rb") as pk_file:
                data_dict = pickle.load(pk_file)

            train_data_dict = data_dict["train"]
            test_data_dict = data_dict["test"]



            train_dataset = CustomizeDataset(train_data_dict, transform=transform)
            test_dataset = CustomizeDataset(test_data_dict, transform=transform)

            # train_dataset = CustomizeDataset(train_data_dict)
            # test_dataset = CustomizeDataset(test_data_dict)

        self.mnist_train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.mnist_test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size)

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
        # print("x: {}, {}".format(x.max(), x.shape))
        # print("train_labels: {}, {}".format(train_labels, train_labels.shape))

        logits = self.forward(x)

        train_prediction = logits.argmax(1)
        train_acc = train_prediction.eq(y).view(-1).float().mean()

        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss, "train_acc": train_acc}
        return {'loss': loss, "train_prediction": train_prediction, "train_y": y, 'log': logs}

    def training_epoch_end(self, outputs):
        epoch_train_loss = torch.stack([x["log"]['train_loss'] for x in outputs]).mean()
        print("\n" + "*" * 10 + "epoch_train_loss: {} ".format(epoch_train_loss))

        epoch_train_pred = torch.cat([x['train_prediction'] for x in outputs], dim=0)
        epoch_train_y = torch.cat([x['train_y'] for x in outputs], dim=0)

        epoch_train_acc = epoch_train_pred.eq(epoch_train_y).view(-1).float().mean()
        print("*" * 10 + "epoch_train_acc : {}\n ".format(epoch_train_acc))

        tensorboard_logs = {'epoch_train_loss': epoch_train_loss, "epoch_train_acc": epoch_train_acc}
        return {'epoch_train_loss': epoch_train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print("x shape: {}".format(x.size()))
        logits = self.forward(x)

        validation_prediction = logits.argmax(1)

        validation_loss = self.cross_entropy_loss(logits, y)
        return {'validation_loss': validation_loss, "validation_prediction": validation_prediction, "validation_y": y}

    def validation_epoch_end(self, outputs):
        epoch_validation_loss = torch.stack([x['validation_loss'] for x in outputs]).mean()
        print("\n" + "*" * 10 + "epoch_validation_loss: {} ".format(epoch_validation_loss))

        epoch_validation_pred = torch.cat([x['validation_prediction'] for x in outputs], dim=0)
        epoch_validation_y = torch.cat([x['validation_y'] for x in outputs], dim=0)
        epoch_validation_acc = epoch_validation_pred.eq(epoch_validation_y).view(-1).float().mean()

        print("*" * 10 + "epoch_validation_acc : {}\n ".format(epoch_validation_acc))

        tensorboard_logs = {'epoch_validation_loss': epoch_validation_loss,
                            "epoch_validation_acc": epoch_validation_acc}
        return {'epoch_validation_loss': epoch_validation_loss, 'log': tensorboard_logs}

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self.forward(x)
    #
    #     test_prediction = logits.argmax(1)
    #
    #     test_loss = self.cross_entropy_loss(logits, y)
    #     return {'test_loss': test_loss, "test_prediction": test_prediction, "test_y": y}
    #
    # def test_epoch_end(self, outputs):
    #     epoch_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     print("\n" + "*" * 10 + "epoch_test_loss: {} ".format(epoch_test_loss))
    #
    #     epoch_test_pred = torch.cat([x['test_prediction'] for x in outputs], dim=0)
    #     epoch_test_y = torch.cat([x['test_y'] for x in outputs], dim=0)
    #     epoch_test_acc = epoch_test_pred.eq(epoch_test_y).view(-1).float().mean()
    #
    #     print("*" * 10 + "epoch_test_acc : {}\n ".format(epoch_test_acc))
    #
    #     tensorboard_logs = {'epoch_test_loss': epoch_test_loss,
    #                         "epoch_test_acc": epoch_test_acc}
    #     return {'epoch_test_loss': epoch_test_loss, 'log': tensorboard_logs}


# train
cuda_index = "0"
args = parameter_setting(cuda_index)
# args.using_torch_dataset = True
args.using_torch_dataset = False


model = LightningMNISTClassifier(args)
trainer = pl.Trainer(gpus=cuda_index, max_epochs=args.epoch_number)

trainer.fit(model)
