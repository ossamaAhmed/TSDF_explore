__author__ = 'Ossama'

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.state_encoder_decoder_v1 import StateEncoderDecoder
from datasets_processing.SDF import SDF
from torchvision.utils import save_image
from tensorboardX import SummaryWriter


class encodingState(object):
    def __init__(self):
        # defining experiment params
        self._num_epochs = 30
        self._dataset = SDF()
        self.tf_writer = SummaryWriter()

    def _to_img(self, x):
        # x = 0.5 * (x+1) #to increase the contrast kind of
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    def train(self):
        model = StateEncoderDecoder().cuda()
        # distance = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        traininig_dataset_length = len(self._dataset.dataloader.dataset)
        for epoch in range(self._num_epochs):
            loss_sum = 0
            for sdf_maps in self._dataset.dataloader:
                sdf_maps = Variable(sdf_maps).cuda()
                sdf_maps = sdf_maps.float()
                # ===================forward=====================
                sdf_maps = sdf_maps[:,:,:,None]
                sdf_maps = sdf_maps.permute(0, 3, 1, 2)
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
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self._num_epochs, loss.data))
                # pic = self._to_img(output.cpu().data)
                # save_image(output.cpu().data, './data/image_gen_{}.png'.format(epoch))
        torch.save(model.state_dict(), './pretrained_models/state_autoencoder_v1.pth')
        self.tf_writer.close()