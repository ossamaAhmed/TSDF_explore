__author__ = 'Ossama'

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.vanilla_ae import Autoencoder
from datasets_processing.MNIST import MNIST
from torchvision.utils import save_image


class encodingState(object):
    def __init__(self):
        # defining experiment params
        self._num_epochs = 100
        self._dataset = MNIST()

    def _to_img(self, x):
        # x = 0.5 * (x+1) #to increase the contrast kind of
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    def train(self):
        model = Autoencoder().cuda()
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        for epoch in range(self._num_epochs):
            for data in self._dataset.dataloader:
                img, _ = data
                img = Variable(img).cuda()
                # ===================forward=====================
                img = img.view(img.shape[0], -1)
                output = model(img)
                loss = distance(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epochs, loss.data))
            if epoch % 5 == 0:
                pic = self._to_img(output.cpu().data)
                save_image(pic, './data/image_gen_{}.png'.format(epoch))
        torch.save(model.state_dict(), './sim_autoencoder.pth')