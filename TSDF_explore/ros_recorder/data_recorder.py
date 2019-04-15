#!/usr/bin/python

import rospy
import numpy as np
import os
from std_msgs.msg import Float32MultiArray


class DataRecorder(object):
    def __init__(self,dataCounter=0):
        self.dataCounter = dataCounter
        try:
            os.mkdir('./submaps')
            os.mkdir('./masks')
        except:
            print('Folder existed! ')

    def mask_generation(self,submap):
        # will change -1000 to be 0 and also generate a mask where 0 is mask
        mask = submap.copy()
        mask[mask!=-1000] = 1
        mask[mask==-1000] = 0
        submap[submap==-1000] = 0
        return submap, mask

    def receive_submap_callback(self, submap):
        submap, mask = self.mask_generation(np.array(submap.data))
        np.savetxt('./submaps/submap'+str(self.dataCounter), submap.reshape((129,129)), delimiter=',')
        np.savetxt('./masks/mask'+str(self.dataCounter), mask.reshape((129,129)), delimiter=',')
        self.dataCounter += 1
        print('Successfully saved submap and mask at step '+str(self.dataCounter-1))


if __name__ == '__main__':
    rospy.init_node('data_recorder.py')
    dr = DataRecorder()
    rospy.Subscriber('/voxblox_rl_simulator/simulation/submap', Float32MultiArray, dr.receive_submap_callback)
    rospy.spin()