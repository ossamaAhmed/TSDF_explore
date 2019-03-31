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
        print(os.getcwd())
        self.training_set = pickle.load(open("./data/SDF/training/pickled/maps", "rb"))
        self.validation_set = pickle.load(open("./data/SDF/validation/pickled/maps", "rb"))
        print("Finished loading data")
        self.dataloader = torch.utils.data.DataLoader(torch.from_numpy(self.training_set), batch_size=5,
                                                      shuffle=True, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(torch.from_numpy(self.validation_set), batch_size=4,
                                                      shuffle=True, num_workers=2)

    def _visualize_dataset(self, start, end):
        for i in range(start, end):
            sns.heatmap(self.training_set[i])
            plt.show()

    def _randomize_unknown_spaces(self):
        raise Exception('Randomize unknown spaces is not yet implemented')

    def _truncate_distances(self):
        raise Exception('Truncate distances is not yet implemented')
