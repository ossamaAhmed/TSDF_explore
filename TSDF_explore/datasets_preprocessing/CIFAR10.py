__author__ = 'Ossama'

import torch
import torchvision.transforms as transforms
import torchvision as tv


class CIFAR10(object):
    def __init__(self):
        self.dataloader = None
        self.testloader = None
        self.classes = None
        self._prepare_data()

    def _prepare_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
        trainTransform = tv.transforms.Compose(
            [tv.transforms.ToTensor(), tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
        trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
        testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)