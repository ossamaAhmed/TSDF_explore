import gym
import numpy as np
import logging.config
from geometry_msgs.msg import Vector3, Quaternion, Transform
import roslaunch
import rospy
from voxblox_rl_simulator.srv import *
from std_srvs.srv import Empty
import os


class RobotEnv(gym.Env):

    def __init__(self):
        self.robot = None
        self.__version__ = "0.1.0"
        self.initialized = False
        self.observations = {}
        self.default_action = [1e-5, 0, 0, 0, 0, 0, 1]
        self.accumulated_observed_voxels = 0
        logging.info("TSDF_explore - Version {}".format(self.__version__))
        self.current_step = -1
        rospy.init_node('voxblox_rl_simulator', anonymous=True)
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [os.path.join(current_dir, "resources/simulator.launch")])
        self.launch.start()
        rospy.loginfo("started the simulator")
        #TODO: Quit logging from ROS
        rospy.wait_for_service('/voxblox_rl_simulator/simulation/move')
        rospy.wait_for_service('/voxblox_rl_simulator/simulation/reset')
        self.do_simulation = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/move', Move)
        self.do_reset = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/reset', Empty)
        #TODO:define action space remember to add restriction of not moving
        return

    def initialize(self):
        pass

    def render(self, mode='human'):
        return

    def set_logging(self, value):
        if value:
            roslaunch.configure_logging(self.uuid)

    def step(self, action):
        if not self.initialized:
            self._take_action(self.default_action)
            self.initialized = True
        else:
            self._take_action(self.default_action)
        self.current_step += 1
        self._take_action(action)
        ob = self._get_obs()
        reward = self._calculate_reward()
        done = self._is_done()
        return ob, reward, done, {}

    def reset(self):
        result = self.do_reset()
        print(result)
        self._take_action(self.default_action)
        return self._get_obs()

    def _take_action(self, action):
        #TODO: double check if the action of type Transform
        #wrap the action into an actual message
        transformed_action = Transform(Vector3(*action[:3]), Quaternion(*action[3:]))
        resp = self.do_simulation(transformed_action)
        self.observations['collsion'] = resp.has_collision.data
        self.observations['submap'] = np.array(resp.submap.data).reshape(resp.submap.layout.dim[0].size,
                                                                         resp.submap.layout.dim[1].size)
        self.observations['newly_observed_voxels'] = resp.newly_observed_voxels.data
        self.accumulated_observed_voxels += self.observations['newly_observed_voxels']
        return

    def _get_obs(self):
        return np.array([self.observations['collsion'],
                         self.observations['submap'],
                         self.observations['newly_observed_voxels']])

    def _calculate_reward(self):
        return self.observations['newly_observed_voxels']

    def _is_done(self):
        return False

    def move(self):
        pass

    def set_dynamics(self):
        pass

    def get_dynamics(self):
        pass

    def set_observations(self):
        pass

    def get_observations(self):
        pass

    def stop_movement(self):
        pass

    def is_stationary(self):
        pass


