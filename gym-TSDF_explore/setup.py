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
from setuptools import setup

setup(name='gym_TSDF_explore',
      version='0.0.1',
      install_requires=['gym',
                        'rospy',
                        'defusedxml',
                        'netifaces',
                        'rospkg',
                        'transforms3d'])