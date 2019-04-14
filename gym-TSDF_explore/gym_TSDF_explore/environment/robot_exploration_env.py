import gym
import numpy as np
import logging.config
from geometry_msgs.msg import Vector3, Quaternion, Transform
import roslaunch
import rospy
from voxblox_rl_simulator.srv import *
from gym_TSDF_explore.environment.robot import Robot
from transforms3d.euler import euler2quat
from std_srvs.srv import Empty
from gym import spaces
import os


class RobotEnv(gym.Env):

    def __init__(self):
        self.robot = None
        self.__version__ = "0.1.0"
        self.initialized = False
        self.observations = {}
        # logging.disabled = True
        # logging.info("TSDF_explore - Version {}".format(self.__version__))
        self.current_step = -1
        rospy.init_node('voxblox_rl_simulator', anonymous=True, log_level=rospy.ERROR)
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        loggers = [name for name in logging.root.manager.loggerDict]
        for logger_name in loggers:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        print(loggers)
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, [os.path.join(current_dir, "resources/simulator.launch")])
        self.launch.start()
        # rospy.loginfo("started the simulator")
        #TODO: Quit logging from ROS
        logging.getLogger('rosout').setLevel(logging.CRITICAL)
        rospy.wait_for_service('/voxblox_rl_simulator/simulation/move')
        rospy.wait_for_service('/voxblox_rl_simulator/simulation/reset')
        self.do_simulation = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/move', Move)
        self.do_reset = rospy.ServiceProxy('/voxblox_rl_simulator/simulation/reset', Empty)
        self.robot = Robot(name='explorer')
        #TODO:define action space remember to add restriction of not moving
        high = np.array([5.0, 5.0, 0.0, np.pi, np.pi, 0])
        low = np.array([-5.0, -5.0, 0.0, -np.pi,-np.pi, 0])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(-1000, 1000, shape=[129, 129], dtype=np.float32)
        return

    def render(self, mode='human'):
        return

    def set_logging(self, value):
        if value:
            roslaunch.configure_logging(self.uuid)

    def step(self, action):
        #TODO:add a check that its not a zero T
        if not self.initialized:
            self._take_action(self.robot.default_action)
            self.initialized = True
        else:
            if np.abs(action[0]) < 1e-5:
                action[0] = 1e-5
            if np.abs(action[1]) < 1e-5:
                action[1] = 1e-5
            self._take_action(action)
        self.current_step += 1
        ob = self._get_obs()
        reward = self._calculate_reward()
        done = self._is_done()
        return ob, reward, done, {}

    def reset(self):
        result = self.do_reset()
        print(result)
        self._take_action(self.robot.default_action)
        return self._get_obs()

    def _take_action(self, action):
        #TODO: double check if the action of type Transform
        #wrap the action into an actual message
        transformed_action = Transform(Vector3(*action[:3]), Quaternion(*euler2quat(*action[3:])))
        resp = self.do_simulation(transformed_action)
        self.observations['collsion'] = resp.has_collision.data
        self.observations['submap'] = np.array(resp.submap.data).reshape(resp.submap.layout.dim[0].size,
                                                                         resp.submap.layout.dim[1].size)
        self.observations['newly_observed_voxels'] = resp.newly_observed_voxels.data
        self.robot.set_observations(self.observations)
        print(self.observations['collsion'])
        return

    def _get_obs(self):
        return np.array(self.observations['submap'])

    def _calculate_reward(self):
        return self.observations['newly_observed_voxels']

    def _is_done(self):
        return self.robot.is_stuck()

    def move(self):
        pass

    def set_dynamics(self):
        pass

    def get_dynamics(self):
        pass

