#!/usr/bin/python

# Script File: pickle_dataset.py
#
# Written By:   Adrian Esser, Alex Millane
# Date Created: 2018-04-29
# Email:        aesser(at)student.ethz.ch
#
# Description:  Pickle training data for faster access!

import numpy as np
import sys
import os
import pickle


def pickle_dataset(input_dir, output_dir):
    # Filenames
    files = os.listdir(input_dir)
    files.sort()
    # Loading the sdfs into a list
    sdf_list = []
    num_files = len(files)
    for i in range(num_files):
        print ('loading map ' + str(i) + '/' + str(num_files))
        file = input_dir + files[i]
        sdf = np.genfromtxt(file, delimiter=',')
        sdf_list.append(sdf)
    sdfs = np.array(sdf_list)
    # Writing to file
    print ('Saving maps')
    filename = 'maps'
    with open(output_dir + filename, 'wb') as fp:
        pickle.dump(sdfs, fp)


# Pickling the dataset
print ('Training Data')
input_data_folder = './output/training/raw/'
output_data_folder = './output/training/pickled/'
pickle_dataset(input_data_folder, output_data_folder)
print ('Validation Data')
input_data_folder = './output/validation/raw/'
output_data_folder = './output/validation/pickled/'
pickle_dataset(input_data_folder, output_data_folder)
