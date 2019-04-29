#!/usr/bin/python
"""
 *!
 * @author    Yimeng
 modified by oahmed
 *
 * Copyright (C) 2019 Autonomous Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.asl.ethz.ch/
 *
 """
import rospy
import numpy as np
import os
from std_msgs.msg import Float32MultiArray
import sys
import roslaunch
import logging
from std_srvs.srv import Empty
from multiprocessing import Process


class DataRecorder(object):
    def __init__(self, map_size, samples_num_train, samples_num_valid):
        rospy.init_node('voxblox_rl_simulator', anonymous=True, log_level=rospy.ERROR)
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid,
                                                       [os.path.join(current_dir, "simulator.launch")])
        self.launch.start()
        rospy.wait_for_service('/voxblox_rl_simulator/simulation/random_walk')
        rospy.wait_for_service('/voxblox_rl_simulator/simulation/reset')
        self.do_random_walk = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/random_walk', Empty)
        self.do_reset = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/reset', Empty)
        self.data_counter = 0
        self.finished_training_data = False
        self.maps_train = np.zeros(shape=[samples_num_train, 2, *map_size])
        self.maps_valid = np.zeros(shape=[samples_num_valid, 2, *map_size])
        self.samples_num_train = samples_num_train
        self.samples_num_valid = samples_num_valid
        self.random_walk_thread = None
        self.alive = True

    def record_data(self):
        self.random_walk_thread = Process(target=self.do_random_walk)
        self.random_walk_thread.start()
        rospy.Subscriber('/voxblox_rl_simulator/simulation/submap', Float32MultiArray, self.receive_submap_callback)
        rospy.spin()

    def mask_generation(self,submap):
        # will change -1000 to be 0 and also generate a mask where 0 is mask
        mask = submap.copy()
        mask[mask!=-1000] = 1
        mask[mask==-1000] = 0
        submap[submap==-1000] = 0
        return submap, mask

    def receive_submap_callback(self, submap):
        new_submap, mask = self.mask_generation(np.array(submap.data))
        if self.finished_training_data:
            self.maps_valid[self.data_counter, 0, :, :] = new_submap.reshape(submap.layout.dim[0].size,
                                                                            submap.layout.dim[1].size)
            self.maps_valid[self.data_counter, 1, :, :] = mask.reshape(submap.layout.dim[0].size,
                                                                            submap.layout.dim[1].size)
        else:
            self.maps_train[self.data_counter, 0, :, :] = new_submap.reshape(submap.layout.dim[0].size,
                                                                         submap.layout.dim[1].size)
            self.maps_train[self.data_counter, 1, :, :] = mask.reshape(submap.layout.dim[0].size,
                                                                        submap.layout.dim[1].size)
        self.data_counter += 1
        print(self.data_counter)
        if not self.finished_training_data and self.data_counter == self.samples_num_train:
            self.data_counter = 0
            self.finished_training_data = True
        elif self.finished_training_data and self.data_counter == self.samples_num_valid:
            #save the files
            current_dir = os.path.dirname(os.path.abspath(__file__))
            os.path.join(current_dir, "../data/random_data_sdf/training")
            np.save(os.path.join(current_dir, "../data/random_data_sdf/training"), self.maps_train)
            np.save(os.path.join(current_dir, "../data/random_data_sdf/validation"), self.maps_valid)
            self.random_walk_thread.terminate()
            rospy.signal_shutdown("Finished")

        if self.data_counter%20 == 0:
            self.random_walk_thread.terminate()
            thread = Process(target=self.do_reset)
            thread.start()
            self.random_walk_thread = Process(target=self.do_random_walk)
            self.random_walk_thread.start()