__author__ = 'Ossama'

import torch
from torch.autograd import Variable
from TSDF_explore.models.state_encoder_decoder_v1 import StateEncoderDecoder
from TSDF_explore.datasets_preprocessing.SDF import SDF
from tensorboardX import SummaryWriter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from TSDF_explore.policies.policy_loader import ModelLoader
import os


class encodingState(object):
    def __init__(self):
        # defining experiment params
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

    def train(self, num_epochs, model_path, gpu=True):
        if gpu:
            model = StateEncoderDecoder().cuda()
        else:
            model = StateEncoderDecoder()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        traininig_dataset_length = len(self._dataset.dataloader.dataset)
        for epoch in range(num_epochs):
            loss_sum = 0
            for sdf_maps in self._dataset.dataloader:
                if gpu:
                    sdf_maps = Variable(sdf_maps).cuda()
                else:
                    sdf_maps = Variable(sdf_maps)
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
            if epoch % 10 == 0:
                self._visualize_dataset(sdf_maps.cpu().data.numpy(), output.cpu().data.numpy())
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,num_epochs, loss.data))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save(model.state_dict(), os.path.join(current_dir, "..", model_path))
        self.tf_writer.close()

    def test(self, model_path, gpu=True):
        model_loader = ModelLoader(model_class="state_encoder_v1",
                                   trained_model_path=model_path,
                                   gpu=gpu)
        model = model_loader.get_inference_model()
        testing_dataset_length = len(self._dataset.testloader.dataset)
        #test now
        validation_loss_sum = 0
        for sdf_maps in self._dataset.testloader:
            if gpu:
                sdf_maps = Variable(sdf_maps, volatile=True).cuda()
            else:
                sdf_maps = Variable(sdf_maps, volatile=True)
            sdf_maps = sdf_maps.float()
            # ===================forward=====================
            output = model(sdf_maps)
            loss = torch.mean(torch.abs((output - sdf_maps) ** 2))
            validation_loss_sum += loss
        print("Final Validation Loss: {:.4f}\n".format(validation_loss_sum/testing_dataset_length))
        return validation_loss_sum/testing_dataset_length