__author__ = 'Ossama'

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.vanilla_ae import Autoencoder
from datasets.CIFAR10 import CIFAR10


class encodingState(object):
    def __init__(self):
        # defining experiment params
        self._num_epochs = 5
        self._dataset = CIFAR10()

    def train(self):
        model = Autoencoder().cpu()
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        for epoch in range(self._num_epochs):
            for data in self._dataset.dataloader:
                img, _ = data
                img = Variable(img).cpu()
                # ===================forward=====================
                output = model(img)
                loss = distance(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epochs, loss.data))
