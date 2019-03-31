# Script File: main.py
#
# Written By:   Adrian Esser
# Date Created: 2018-03-19 
# Email:        aesser(at)student.ethz.ch
#
# Description: This is the first iteration of a Convolutional Auto Encoder (CAE)
#              for generating compact and informative representations of 2D 
#              SDF map segments, which we hope to eventually use for registering 
#              local submaps to a global map. 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import shutil
from sklearn.model_selection import KFold # for generating training batches 
import pickle

###########################################
#                                         #
# 1) Load in all of the training data!    #
#                                         #
###########################################

# New way to load pickled data
path_to_training_data = './p_dataset/training_data'
path_to_validation_data = './p_dataset/validation_data'

print("Loading Map Training Data...")
with open(path_to_training_data, 'rb') as fp:
    training_data = pickle.load(fp)
print("Finished Loading Training Data...")

print("Loading Map Validation Data...")
with open(path_to_validation_data, 'rb') as fp:
    validation_data = pickle.load(fp)
print("Finished Loading Validation Data...")

print(np.shape(training_data))
print(training_data[0])

sys.exit()

# Old way to load data
'''
path_to_training_data = './dataset_debug/training_data/'
path_to_validation_data = './dataset_debug/validation_data/'

print("Loading Map Training Data...")
training_data = np.array([np.genfromtxt(path_to_training_data+m_file, delimiter=',') for m_file in
        os.listdir(path_to_training_data)], dtype=np.float64)
print("Finished Loading Training Data...")

print("Loading Map Validation Data...")
validation_data = np.array([np.genfromtxt(path_to_validation_data+m_file, delimiter=',') for m_file
        in os.listdir(path_to_validation_data)], dtype=np.float64)
print("Finished Loading Validation Data...")
'''

n = np.shape(training_data)[0] # number of maps for training
w = np.shape(training_data)[1] # side length of maps (maps will always be square)

# TODO: Give better name
n2 = np.shape(validation_data)[0] # number of maps for validation 
###########################################
#                                         #
#  2)     Define CAE Structure!           #
#                                         #
###########################################

# Inspired From: https://towardsdatascience.com/autoencoders-introduction-and-implementation-3f40483b0a85

learning_rate = 0.0001
inputs_ = tf.placeholder(tf.float32, (None, w, w, 1), name='inputs') # for now w = 64
targets_ = tf.placeholder(tf.float32, (None, w, w, 1), name='targets')
do_dropout = tf.placeholder(tf.bool, (), name='do_dropout')

### Encoder
k1 = int(7) # NOTE: kernel size (please make this odd)
p1 = int((k1-1)/2) # padding size
padded1 = tf.pad(inputs_, tf.constant([[0,0],[p1,p1],[p1,p1],[0,0]]), "SYMMETRIC")
conv1 = tf.layers.conv2d(inputs=padded1, filters=16, kernel_size=(k1,k1), padding='VALID',
        activation=tf.nn.relu)
# Now 128x128x16

maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding="VALID")
# Now 64x64x16

k2 = int(5)
p2 = int((k2-1)/2)
padded2 = tf.pad(maxpool1, tf.constant([[0,0],[p2,p2],[p2,p2],[0,0]]), "SYMMETRIC")
conv2 = tf.layers.conv2d(inputs=padded2, filters=32, kernel_size=(k2,k2), padding="VALID", activation=tf.nn.relu)
# Now 64x64x32

maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding="VALID")
# Now 32x32x32

k3 = int(3)
p3 = int((k3-1)/2)
padded3 = tf.pad(maxpool2, tf.constant([[0,0],[p3,p3],[p3,p3],[0,0]]), "SYMMETRIC")
conv3 = tf.layers.conv2d(inputs=padded3, filters=16, kernel_size=(k3,k3), padding="VALID", activation=tf.nn.relu)
# Now 32x32x16

maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding="VALID")
# Now 16x16x16

k4 = int(3)
p4 = int((k4-1)/2)
padded4 = tf.pad(maxpool3, tf.constant([[0,0],[p4,p4],[p4,p4],[0,0]]), "SYMMETRIC")
conv4 = tf.layers.conv2d(inputs=padded4, filters=16, kernel_size=(k4,k4), padding="VALID", activation=tf.nn.relu)
# Now 16x16x16

maxpool4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=(2,2), padding="VALID")
# Now 8x8x16

maxpool4_to_flat = tf.reshape(maxpool4, [-1, 8*8*16]) # is there a way to make this automatic? 
# Now 1x1024

# Link for Dropout layers: https://www.tensorflow.org/versions/master/tutorials/layers
dropout1 = tf.layers.dropout(inputs=maxpool4_to_flat, rate=0.1, training=do_dropout)
fc1 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
# Now 1x512

dropout2 = tf.layers.dropout(inputs=fc1, rate=0.1, training=do_dropout)
encoded = tf.layers.dense(inputs=fc1, units=256, activation=tf.nn.relu)
# Now 1x256

# The final encoded image will be a vector of length 256.
# This yields a compression of approximately 1.5%  (100% * 256 / (128*128))

dropout3 = tf.layers.dropout(inputs=encoded, rate=0.05, training=do_dropout)

### Decoder
fc2 = tf.layers.dense(inputs=dropout3, units=512, activation=tf.nn.relu)
# Now 1x512

dropout4 = tf.layers.dropout(inputs=fc2, rate=0.1, training=do_dropout)

fc3 = tf.layers.dense(inputs=dropout4, units=1024, activation=tf.nn.relu)
# Now 1x1024

fc3_to_mat = tf.reshape(fc3, [-1, 8, 8, 16])
# Now 8x8x16

upsample1 = tf.image.resize_images(fc3_to_mat, size=(16,16), method=tf.image.ResizeMethod.BILINEAR)
# Now 16x16x16

k5 = int(3)
p5 = int((k5-1)/2)
padded5 = tf.pad(upsample1, tf.constant([[0,0],[p5,p5],[p5,p5],[0,0]]), "SYMMETRIC")
conv5 = tf.layers.conv2d(inputs=padded4, filters=16, kernel_size=(k5,k5), padding="VALID", activation=tf.nn.relu)
# Now 16x16x16

upsample2 = tf.image.resize_images(conv5, size=(32,32), method=tf.image.ResizeMethod.BILINEAR)
# Now 32x32x16

k6 = int(3)
p6 = int((k6-1)/2)
padded6 = tf.pad(upsample2, tf.constant([[0,0],[p6,p6],[p6,p6],[0,0]]), "SYMMETRIC")
conv6 = tf.layers.conv2d(inputs=padded6, filters=16, kernel_size=(k6,k6), padding="VALID", activation=tf.nn.relu)
# Now 32x32x16

upsample3 = tf.image.resize_images(conv6, size=(64,64), method=tf.image.ResizeMethod.BILINEAR)
# Now 64x64x16

k7 = int(5)
p7 = int((k7-1)/2)
padded7 = tf.pad(upsample3, tf.constant([[0,0],[p7,p7],[p7,p7],[0,0]]), "SYMMETRIC")
conv7 = tf.layers.conv2d(inputs=padded7, filters=32, kernel_size=(k7,k7), padding="VALID", activation=tf.nn.relu)
# Now 64x64x32

upsample4 = tf.image.resize_images(conv7, size=(128,128), method=tf.image.ResizeMethod.BILINEAR)
# Now 128x128x32

k8 = int(7)
p8 = int((k8-1)/2)
padded8 = tf.pad(upsample4, tf.constant([[0,0],[p8,p8],[p8,p8],[0,0]]), "SYMMETRIC")
conv8 = tf.layers.conv2d(inputs=padded8, filters=16, kernel_size=(k8,k8), padding="VALID", activation=tf.nn.relu)
# Now 128x128x16

k9 = int(7)
p9 = int((k9-1)/2)
padded9 = tf.pad(conv8, tf.constant([[0,0],[p9,p9],[p9,p9],[0,0]]), "SYMMETRIC")
decoded = tf.layers.conv2d(inputs=padded9, filters=1, kernel_size=(k9,k9), padding="VALID", activation=tf.nn.relu)
# Now 128x128x1

# We are not going to pass the final layer through a sigmoid (for now) because I'm
# not sure how we are going to do data normalization yet... We should talk about this, because
# typically normalization really really helps the network converge faster (keeps everything on the
# same scale).

# Compute L2-norm between decoded image and original image (no normalization)
loss_L2 = tf.square(tf.subtract(decoded, inputs_))
loss_L1 = tf.abs(tf.subtract(decoded, inputs_))

cost = tf.reduce_mean(loss_L2) # Should be removing all dimensions by default
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

###########################################
#                                         #
#  3)  Init. Network and Training Params  #
#                                         #
###########################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 1
batch_size = 5
splits = int(np.floor(n/batch_size))

training_cost_vec = []
validation_cost_vec = []
epoch_vec_train = []
epoch_vec_validate = []

# Define the saver object
saver = tf.train.Saver()
tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inputs_)
tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, encoded)
tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, decoded)
tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, do_dropout)

# Validation images (ready for network)
imgs_val = validation_data.reshape((-1, w, w, 1))

print(np.shape(validation_data))
print(np.shape(imgs_val))

print(np.shape(training_data))


# Train the network
for e in range(epochs):
    kf = KFold(n_splits=int(np.floor(n/batch_size)), shuffle=True)
    i = 0

    # At the beginning of each epoch, test all of the validation data in one shot
    validation_cost = sess.run([cost], feed_dict={inputs_:imgs_val, do_dropout:False})[0]
    validation_cost_vec.append(validation_cost)
    epoch_vec_validate.append(e+1)
    print("Epoch: {}/{}...".format(e+1, epochs), "Validation Loss: {:.4f}\n".format(validation_cost))

    for _ , t_idx in kf.split(training_data): # Using KFolds for batch generation... Better way using TF maybe?
        i += 1

        #print(t_idx)
        #print(np.shape(training_data[t_idx].reshape((-1,w,w,1))))
        #sys.exit()

        imgs = training_data[t_idx].reshape((-1, w, w, 1)) # reshape for TF
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_:imgs, do_dropout:False})
        training_cost_vec.append(batch_cost) 
        epoch_vec_train.append(e+1+(i-1)/splits)

        #decoded_out = sess.run([decoded], feed_dict={inputs_:imgs})
        #print(np.shape(decoded_out))

        #encoded_out = sess.run([encoded], feed_dict={inputs_:imgs})
        #print(np.shape(encoded_out))
        
        print("Epoch: {}/{}...".format(e+1, epochs), "Training loss: {:.4f}".format(batch_cost))

# Final validationn cost
validation_cost, _ = sess.run([cost, opt], feed_dict={inputs_:imgs_val, do_dropout:False})
validation_cost_vec.append(validation_cost)
epoch_vec_validate.append(epochs+1)
print("Final Validation Loss: {:.4f}\n".format(validation_cost))

# Save the graph. 
# TODO: Make this nicer (variable for path and model name, checkpoint saving 
#       during training (see PDF on experiment management in Dropbox)
if os.path.exists("trained_model"):
    shutil.rmtree("trained_model")
os.mkdir("trained_model")
saver.save(sess, './trained_model/model')

# Save the training loss results to file.
output_training = np.concatenate((np.matrix(epoch_vec_train).T, np.matrix(training_cost_vec).T), axis=1)
np.savetxt("trained_model/training_loss_results.csv", output_training, fmt='%.10f', delimiter=',', newline='\n', comments='')

# Save the validation loss results to file.
output_validation = np.concatenate((np.matrix(epoch_vec_validate).T, np.matrix(validation_cost_vec).T), axis=1)
np.savetxt("trained_model/validation_loss_results.csv", output_validation, fmt='%.10f', delimiter=',', newline='\n', comments='')

# Plot!
plt.plot(epoch_vec_train, training_cost_vec, 'b')
plt.plot(epoch_vec_validate, validation_cost_vec, 'm')
plt.xlabel('Epoch')
plt.ylabel('Batch Losses')
plt.title('Training Curve')
plt.legend(['Training Loss', 'Validation Loss'])
plt.savefig("trained_model/training_loss_graph.pdf", bbox_inches='tight')
plt.show()




