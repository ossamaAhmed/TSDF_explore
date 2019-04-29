"""
 *!
 * @author    Ossama Ahmed
 * @email     oahmed@ethz.ch
 *
 * Copyright (C) 2019 Autonomous Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.asl.ethz.ch/
 *
 """
import torch
from torch.autograd import Variable
import torch.nn as nn
from TSDF_explore.models.state_encoder_decoder_v1 import StateEncoderDecoder
from TSDF_explore.datasets_preprocessing.SDF import SDF
from tensorboardX import SummaryWriter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from TSDF_explore.policies.policy_loader import ModelLoader
import os
import warnings


class EncodingState(object):
    def __init__(self, do_randomize_unknown_spaces=False,
                 prepare_random_data=False, prepare_data_from_matlab=False, prepare_data_from_sim=False):
        # defining experiment params
        self._dataset = SDF(do_randomize_unknown_spaces=do_randomize_unknown_spaces,
                            prepare_random_data=prepare_random_data, prepare_data_from_matlab=prepare_data_from_matlab,
                            prepare_data_from_sim=prepare_data_from_sim)
        self.tf_writer = SummaryWriter()
        warnings.filterwarnings("ignore")

    def _convert_plot_to_image(self, figure):
        figure.canvas.draw()
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data

    def _visualize_dataset(self, original_input, decoded_output, epoch):
        #choose a random image
        random_im_index = np.random.randint(0, decoded_output.shape[0])
        test_map = original_input[random_im_index]
        output = decoded_output[random_im_index]
        norm_input = np.max(test_map)
        norm_output = np.max(output)
        norm = np.max([norm_input, norm_output])
        #plotting
        fig = plt.figure(figsize=(30, 10))
        ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0))
        ax1.set_title('original_input')
        sns.heatmap(test_map[0, :, :]/norm, ax=ax1)
        ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 1))
        ax2.set_title('decoded_result')
        sns.heatmap(output[0, :, :] / norm, ax=ax2)
        ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 0))
        ax3.set_title('original_input_second_channel')
        sns.heatmap(test_map[1, :, :], ax=ax3)
        ax4 = plt.subplot2grid(shape=(2, 2), loc=(1, 1))
        ax4.set_title('decoded_result_second_channel')
        sns.heatmap(output[1, :, :], ax=ax4)
        # plt.show()
        self.tf_writer.add_image("intermediate_results", self._convert_plot_to_image(figure=fig),
                                 global_step=epoch, dataformats='HWC')
        return

    def train(self, num_epochs, model_path, gpu=True):
        if gpu:
            model = StateEncoderDecoder().cuda()
        else:
            model = StateEncoderDecoder()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
        traininig_dataset_length = len(self._dataset.dataloader.dataset)
        distance = nn.MSELoss()
        for epoch in range(num_epochs):
            loss_sum = 0
            for sdf_maps in self._dataset.dataloader:
                sdf_maps = sdf_maps[:, :, 0:-1, 0:-1]
                if gpu:
                    sdf_maps = Variable(sdf_maps).cuda()
                else:
                    sdf_maps = Variable(sdf_maps)
                sdf_maps = sdf_maps.float()
                # ===================forward=====================
                output = model(sdf_maps)
                loss = distance(output, sdf_maps)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss
            # ===================log========================
            self.tf_writer.add_scalar('data/loss', loss_sum / traininig_dataset_length, epoch)
            if epoch % 10 == 0:
                self._visualize_dataset(sdf_maps.cpu().data.numpy(), output.cpu().data.numpy(), epoch=epoch)
                print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1,num_epochs, loss.data))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save(model.state_dict(), os.path.join(current_dir, "..", model_path))
        self.tf_writer.close()

    #TODO:this function is not giving the same accuracy every time
    def test(self, model_path, gpu=True):
        model_loader = ModelLoader(model_class="state_encoder_v1",
                                   trained_model_path=model_path,
                                   gpu=gpu)
        model = model_loader.get_inference_model()
        testing_dataset_length = len(self._dataset.testloader.dataset)
        #test now
        validation_loss_sum = 0
        distance = nn.MSELoss()
        for sdf_maps in self._dataset.testloader:
            sdf_maps = sdf_maps[:, :, 0:-1, 0:-1]
            if gpu:
                sdf_maps = Variable(sdf_maps).cuda()
            else:
                sdf_maps = Variable(sdf_maps)
            sdf_maps = sdf_maps.float()
            # ===================forward=====================
            with torch.no_grad():
                output = model(sdf_maps)
                loss = distance(output, sdf_maps)
                print("loss", loss)
                validation_loss_sum += loss
        print(validation_loss_sum)
        print("Final Validation Loss: {:.4f}\n".format(validation_loss_sum/testing_dataset_length))
        return validation_loss_sum/testing_dataset_length