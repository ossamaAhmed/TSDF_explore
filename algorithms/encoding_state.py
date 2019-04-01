__author__ = 'Ossama'

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.state_encoder_decoder_v1 import StateEncoderDecoder
from datasets_processing.SDF import SDF
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class encodingState(object):
    def __init__(self):
        # defining experiment params
        self._num_epochs = 30
        self._dataset = SDF()
        self.tf_writer = SummaryWriter()

    def _visualize_dataset(self, original_input, decoded_output):
        #choose a random image
        random_im_index = np.random.randint(0, decoded_output.shape[0])
        test_map = original_input[random_im_index]
        output = decoded_output[random_im_index]
        norm_input = np.max(test_map)
        norm_output = np.max(output)
        norm = np.max([norm_input, norm_output])
        sns.heatmap(test_map[0, :, :]/norm)
        plt.show()
        sns.heatmap(output[0, :, :] / norm)
        plt.show()

    def train(self):
        model = StateEncoderDecoder().cuda()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        traininig_dataset_length = len(self._dataset.dataloader.dataset)
        for epoch in range(self._num_epochs):
            loss_sum = 0
            for sdf_maps in self._dataset.dataloader:
                sdf_maps = Variable(sdf_maps).cuda()
                sdf_maps = sdf_maps.float()
                # ===================forward=====================
                output = model(sdf_maps)
                loss = torch.mean(torch.abs((output - sdf_maps)**2))
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss
            # ===================log========================
            self.tf_writer.add_scalar('data/loss', loss_sum / traininig_dataset_length, epoch)
            if epoch % 5 == 0:
                self._visualize_dataset(sdf_maps.cpu().data.numpy(), output.cpu().data.numpy())
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epochs, loss.data))
        torch.save(model.state_dict(), './pretrained_models/state_autoencoder_v1.pth')
        self.tf_writer.close()