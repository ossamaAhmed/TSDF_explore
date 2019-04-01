import torch
import torchvision.transforms as transforms
import torchvision as tv
import pickle
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class SDF(object):
    def __init__(self):
        self.dataloader = None
        self.testloader = None
        self.training_set = None
        self.validation_set = None
        self._prepare_data()

    def _prepare_data(self):
        print("Loading data now")
        self.training_set = pickle.load(open("./data/SDF/training/pickled/maps", "rb"))
        self.validation_set = pickle.load(open("./data/SDF/validation/pickled/maps", "rb"))
        self.training_set = self._randomize_unknown_spaces(self.training_set)
        self.validation_set = self._randomize_unknown_spaces(self.validation_set)
        print("Finished loading data")
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.training_set), batch_size=5,
                                                      shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(torch.from_numpy(self.validation_set), batch_size=4,
                                                      shuffle=True, num_workers=2)

    def _visualize_dataset(self, start, end):
        for i in range(start, end):
            sns.heatmap(self.training_set[i])
            plt.show()

    def _randomize_unknown_spaces(self, data):
        output = np.expand_dims(data, axis=1)
        unknows_space_indicator = np.ones(shape=output.shape, dtype=np.float32)
        map_width = output.shape[2]
        map_length = output.shape[3]
        number_of_unknown_blobs = 10
        for i in range(number_of_unknown_blobs):
            row_indicies = np.random.randint(0, map_width, size=(2,))
            col_indicies = np.random.randint(0, map_length, size=(2,))
            row_indicies.sort()
            col_indicies.sort()
            unknows_space_indicator[:, 0, row_indicies[0]:row_indicies[1], col_indicies[0]:col_indicies[1]] = 0
        output = np.concatenate((output, unknows_space_indicator), axis=1)
        return output

    def _truncate_distances(self):
        raise Exception('Truncate distances is not yet implemented')
