__author__ = 'Yimeng'

import numpy as np
import pickle 
import seaborn as sns 
import matplotlib.pyplot as plt
import torch

class DataLoader():
    def __init__(self): 
        self.load_data()
        self.check_dataset()

    def load_data(self):
        # New way to load pickled data
        # path_to_training_data = './sdf_generation/output/training/pickled/maps_100'
        # path_to_training_mask = './sdf_generation/output/training/pickled/mask_100'
        # path_to_validation_data = './sdf_generation/output/validation/pickled/maps'
        # path_to_validation_mask = './sdf_generation/output/validation/pickled/mask'
        path_to_training_data = './sdf_generation/output/training/pickled/rl_map_train'
        path_to_training_mask = './sdf_generation/output/training/pickled/rl_mask_train'
        path_to_validation_data = './sdf_generation/output/validation/pickled/rl_map_valid'
        path_to_validation_mask = './sdf_generation/output/validation/pickled/rl_mask_valid'

        print("Loading Map Training Data...")
        self.training_data = pickle.load(open(path_to_training_data, 'rb'))
        print("Finished Loading Training Data...")

        print("Loading Mask Data...")
        self.training_mask = pickle.load(open(path_to_training_mask, 'rb'))
        print("Finished Loading Mask Data...")
        
        print("Processing Masked Data...")
        self.training_data = self.mask_map(self.training_data,self.training_mask)

        print("Loading Map Validation Data...")
        self.validation_data = pickle.load(open(path_to_validation_data, 'rb'))
        print("Finished Loading Validation Data...")

        print("Loading Mask Data...")
        self.validation_mask = pickle.load(open(path_to_validation_mask, 'rb'))
        print("Finished Loading Mask Data...")
        
        print("Processing Masked Data...")
        self.validation_data = self.mask_map(self.validation_data,self.validation_mask)

    def check_dataset(self):
        for i in reversed(range(self.training_data.shape[0])):
            if self.training_data[i].size < 4:
                self.training_data = np.delete(self.training_data,i)
            
    def visualise_dataset(self, start, end):
        for i in range(start, end):
            sns.heatmap(self.training_data[i])
            plt.show()

    def mask_map(self, _map, _mask):
        output = np.expand_dims(_map, axis=1)
        output_mask = np.expand_dims(_mask, axis=1)
        output = np.concatenate([output, output_mask], axis=1)
        return output
