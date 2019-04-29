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
import torchvision.transforms as transforms
import torchvision as tv
import pickle
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class SDF(object):
    def __init__(self, do_randomize_unknown_spaces=False,
                 prepare_random_data=False, prepare_data_from_matlab=False, prepare_data_from_sim=False):
        self.dataloader = None
        self.testloader = None
        self.training_set = None
        self.validation_set = None
        self.do_randomize_unknown_spaces = do_randomize_unknown_spaces
        self.prepare_data_from_sim = prepare_data_from_sim
        if prepare_random_data:
            self._prepare_data_random()
        else:
            self._prepare_data()

    def _prepare_data(self):
        print("Loading data now")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if self.do_randomize_unknown_spaces:
            self.training_set = pickle.load(open(os.path.join(current_dir, "../data/SDF/training/pickled/maps"), "rb"))
            self.validation_set = pickle.load(
                open(os.path.join(current_dir, "../data/SDF/validation/pickled/maps"), "rb"))
            self.training_set = self._randomize_unknown_spaces(self.training_set)
            self.validation_set = self._randomize_unknown_spaces(self.validation_set)
        elif self.prepare_data_from_sim:
            self.training_set = np.load(os.path.join(current_dir, "../data/random_data_sdf/training.npy"))
            self.validation_set = np.load(os.path.join(current_dir, "../data/random_data_sdf/validation.npy"))
        else:
            self.training_set = pickle.load(open(os.path.join(current_dir, "../data/SDF/training/pickled/maps"), "rb"))
            self.validation_set = pickle.load(
                open(os.path.join(current_dir, "../data/SDF/validation/pickled/maps"), "rb"))
            self.training_set = self._add_unknown_spaces_channel(self.training_set)
            self.validation_set = self._add_unknown_spaces_channel(self.validation_set)
        print("Finished loading data")
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.training_set), batch_size=5,
                                                      shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(torch.from_numpy(self.validation_set), batch_size=64,
                                                      shuffle=False, num_workers=4)

    #rl_map_valid
    def _prepare_data_random(self):
        print("Loading data now")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.training_set = pickle.load(open(os.path.join(current_dir, "../data/SDF/training/pickled/rl_map_train"), "rb"))
        self.training_set_mask = pickle.load(
            open(os.path.join(current_dir, "../data/SDF/training/pickled/rl_mask_train"), "rb"))
        self.validation_set = pickle.load(
            open(os.path.join(current_dir, "../data/SDF/validation/pickled/rl_map_valid"), "rb"))
        self.validation_set_mask = pickle.load(
            open(os.path.join(current_dir, "../data/SDF/validation/pickled/rl_mask_valid"), "rb"))
        print("Finished loading data")
        print("Processing Masked Data...")
        self.training_data = self.mask_map(self.training_set, self.training_set_mask)
        print("Processing Masked Data...")
        self.validation_data = self.mask_map(self.validation_set, self.validation_set_mask)
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.training_data), batch_size=5,
                                                      shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(torch.from_numpy(self.validation_data), batch_size=64,
                                                      shuffle=False, num_workers=4)

    def mask_map(self, _map, _mask):
        output = np.expand_dims(_map, axis=1)
        output_mask = np.expand_dims(_mask, axis=1)
        output = np.concatenate([output, output_mask], axis=1)
        return output

    def _visualize_dataset(self, start, end):
        for i in range(start, end):
            sns.heatmap(self.training_set[i])
            plt.show()

    def _randomize_unknown_spaces(self, data):
        output = np.expand_dims(data, axis=1)
        unknown_space_indicator = np.ones(shape=output.shape, dtype=np.float32)
        map_width = output.shape[2]
        map_length = output.shape[3]
        number_of_unknown_blobs = 10
        for i in range(number_of_unknown_blobs):
            row_indicies = np.random.randint(0, map_width, size=(2,))
            col_indicies = np.random.randint(0, map_length, size=(2,))
            row_indicies.sort()
            col_indicies.sort()
            unknown_space_indicator[:, 0, row_indicies[0]:row_indicies[1], col_indicies[0]:col_indicies[1]] = 0
            output[:, 0, row_indicies[0]:row_indicies[1], col_indicies[0]:col_indicies[1]] = 0
        output = np.concatenate((output, unknown_space_indicator), axis=1)
        return output[:, :, :128, :128]

    def _add_unknown_spaces_channel(self, data):
        output = np.expand_dims(data, axis=1)
        unknown_space_indicator = np.ones(shape=output.shape, dtype=np.float32)
        output = np.concatenate((output, unknown_space_indicator), axis=1)
        return output

    def _truncate_distances(self):
        raise Exception('Truncate distances is not yet implemented')
