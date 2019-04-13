__author__ = 'Ossama'

import torch
import torchvision.transforms as transforms
import torchvision as tv


class MNIST(object):
    def __init__(self):
        self.dataloader = None
        self.testloader = None
        self.classes = None
        self._prepare_data()

    def _prepare_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        trainset = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
        testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)