# Script File: verify_trained_model.py
#
# Written By:   Adrian Esser
# Date Created: 2018-03-19
# Email:        aesser(at)student.ethz.ch
#
# Description:  This file loads a trained model and compares the inputs and outputs
#               of the CAE (on the trained data for now). This is mostly a sanity check
#               to see if everything is working...

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import *
import sys
import os
import pickle

from PIL import Image

###########################################
#                                         #
# 1) Load the trained network             #
#                                         #
###########################################
sess = tf.Session()

exp_folder = './trained_model/'
exp_name = ''

saver = tf.train.import_meta_graph(exp_folder+exp_name+'model.meta')
saver.restore(sess, tf.train.latest_checkpoint(exp_folder+exp_name))

###########################################
#                                         #
# 2) Grab the submaps for testing         #
#                                         #
###########################################
path_to_data = './p_dataset/validation_data'

print("Loading Map Training Data...")
with open(path_to_data, 'rb') as fp:
        data = pickle.load(fp)
print("Finished Loading Training Data...")


# Validation Set
#test_map = data[20,:,:] # multiple blobs 
test_map = data[5,:,:] # closely spaced features
#test_map = data[25,:,:] # freaky ass shit
#test_map = data[35,:,:] # weird corner effect (top left)
#test_map = data[39,:,:] # complete disaster

# Test Set
#test_map = data[20,:,:] 
#test_map = data[5,:,:] # closely spaced features
#test_map = data[25,:,:] # ALSO PRETTY GOOD! 
#test_map = data[35,:,:] # THIS ONE IS GOOD!
#test_map = data[39,:,:] # gets separation quite well

norm = np.max(np.max(test_map))

#img_in = Image.fromarray(np.uint8(cm.jet(test_map/norm)*255))
#img_in_larger = img_in.resize((512, 512), Image.BICUBIC)

###########################################
#                                         #
# 3) Run image through CAE and get output #
#                                         #
###########################################
map_input = test_map.reshape(1, 128, 128, 1)

all_vars = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
inputs_ = all_vars[0] # cannot think of a smarter way to do this right now
encoded = all_vars[1]
decoded = all_vars[2]
do_dropout = all_vars[3]

output= sess.run([decoded], feed_dict={inputs_:map_input, do_dropout:False})[0]
output = output.reshape(128,128)

###########################################
#                                         #
# 4) Plotting!                            #
#                                         #
###########################################
norm_input = np.max(test_map)
norm_output = np.max(output)
norm = np.max([norm_input, norm_output])

fig, ax = plt.subplots(1, 2)

print(ax)

ax0 = ax[0]
ax0.imshow(test_map/norm)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.set_xlabel('Input Map', fontsize=18)

ax0 = ax[1]
ax0.imshow(output/norm)
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.set_xlabel('Decoded Map', fontsize=18)

plt.suptitle('Comparison between Input and Decoded Map', y=0.9, fontsize=24)
plt.savefig("CAE_comparison.pdf", format="pdf", bbox_inches = 'tight')

plt.show()

 
