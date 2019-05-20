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
import gym
import numpy as np
import logging.config
from geometry_msgs.msg import Vector3, Quaternion, Transform
import roslaunch
import rospy
from voxblox_rl_simulator.srv import *
from gym_TSDF_explore.environment.robot import Robot
from transforms3d.euler import euler2quat, quat2euler
from std_srvs.srv import Empty
from gym import spaces
import os
import shutil
import torch
from torch.autograd import Variable
from std_msgs.msg import Float32MultiArray
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
import time


class RobotEnv(gym.Env):

    def __init__(self):
        self.robot = None
        self.__version__ = "0.1.0"
        self.observations = {}
        # logging.disabled = True
        logging.info("TSDF_explore - Version {}".format(self.__version__))
        self.current_step = 0
        self.global_step = 0
        rospy.init_node('voxblox_rl_simulator', anonymous=True, log_level=rospy.ERROR)
        self.robot = Robot(name='explorer')
        #TODO:define action space remember to add restriction of not moving
        self.action_scaling_factors = np.array([10.0, 10.0, np.pi])
        self.high = np.array([10.0, 10.0, np.pi])
        self.low = np.array([-10.0, -10.0, -np.pi])
        self.esdf_map_length = 128
        self.action_space = spaces.Box(low=self.low / self.action_scaling_factors, high=self.high / self.action_scaling_factors, dtype=np.float32)
        self.observation_space = spaces.Box(-1000, 1000, shape=[self.esdf_map_length, self.esdf_map_length, 1], dtype=np.float32)
        self.observations_encoder = None
        self.set_gpu_on = False
        self.episode_num = -1
        self.world_map = None
        self.observed_world_map = None
        self.current_location = None
        self.save_trajectory_info = False
        self.tf_writer = None
        self.overfit_to_one_map = False
        self.namespace = None
        self.logging_iter = 50
        self.trajectory_info_dir = None
        self.visualize_plots = True
        self.validation_mode = False

        #logging stuff variables
        self.location_history = []
        self.path = []
        self.executed_actions_history = []
        self.actions_history = []

        self.env_location_history = []
        self.env_path = []
        self.env_executed_actions_history = []
        self.env_actions_history = []
        self.env_maps_history = []
        self.env_observed_maps_history = []
        loggers = [name for name in logging.root.manager.loggerDict]
        for logger_name in loggers:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        return

    def initialize_environment(self, name_space, tf_writer,
                               trajectory_info_dir,
                               save_trajectory_info=False, visualize_plots=False):
        name_space = str(name_space)
        self.namespace = name_space
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cli_args = [os.path.join(current_dir, "resources/simulator_2D.launch"), 'namespace:=' + name_space]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        self.launch = roslaunch.parent.ROSLaunchParent(self.uuid, roslaunch_file)
        self.launch.start()
        # rospy.loginfo("started the simulator")
        # TODO: Quit logging from ROS
        logging.getLogger('rosout').setLevel(logging.CRITICAL)
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/move')
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/reset')
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/reset_observed_map')
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/reset_robot_state')
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/set_random_starting_robot_pose')
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/get_world_map')
        rospy.wait_for_service("/" + name_space + '/voxblox_rl_simulator/simulation/get_observed_world_map')
        self.do_simulation = rospy.ServiceProxy("/" + name_space + '/voxblox_rl_simulator/simulation/move', Move)
        self.do_reset = rospy.ServiceProxy("/" + name_space + '/voxblox_rl_simulator/simulation/reset', Empty)
        self.do_reset_observed_map = rospy.ServiceProxy(
            "/" + name_space + '/voxblox_rl_simulator/simulation/reset_observed_map', Empty)
        self.do_reset_state = rospy.ServiceProxy(
            "/" + name_space + '/voxblox_rl_simulator/simulation/reset_robot_state', Empty)
        self.do_set_random_starting_pose = rospy.ServiceProxy(
            "/" + name_space + '/voxblox_rl_simulator/simulation/set_random_starting_robot_pose', Empty)
        self.get_world_map = rospy.ServiceProxy(
            "/" + name_space + '/voxblox_rl_simulator/simulation/get_world_map', Empty)
        self.get_observed_world_map = rospy.ServiceProxy(
            "/" + name_space + '/voxblox_rl_simulator/simulation/get_observed_world_map', Empty)
        rospy.Subscriber("/" + name_space + '/voxblox_rl_simulator/observed_worldmap', Float32MultiArray,
                         self.get_observed_world_map_topic_subscriber)
        rospy.Subscriber("/" + name_space + '/voxblox_rl_simulator/worldmap', Float32MultiArray,
                         self.get_world_map_topic_subscriber)
        rospy.Subscriber("/" + name_space + '/voxblox_rl_simulator/worldmap', Float32MultiArray,
                         self.get_observed_world_map_topic_subscriber)
        self.trajectory_info_dir = trajectory_info_dir
        self.tf_writer = tf_writer
        self.save_trajectory_info = save_trajectory_info
        self.visualize_plots = visualize_plots

    def _convert_plot_to_image(self, figure):
        figure.canvas.draw()
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data

    def _visualize_world_map(self):
        # choose a random image
        self.path = np.array(self.path)
        self.location_history = np.array(self.location_history)
        self.actions_history = np.array(self.actions_history)
        # plotting
        # fig = plt.figure()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 50))
        ax.set_title('world_map')
        sns.heatmap(self.world_map[0, :, :], ax=ax)
        # sns.scatterplot(x=[30, 200],
        #                 y=[60, 500],
        #                 marker='*',
        #                 color='yellow',
        #                 s=500,
        #                 ax=ax2)
        # sns.scatterplot(x=[self.path[0, 0]],
        #                  y=[self.path[0, 1]],
        #                  marker='*',
        #                  color='yellow',
        #                  s=500,
        #                  ax=ax2)
        # ax = result.ax_heatmap  # this is the important part
        # ax1.scatter([30, 30], [30, 60], marker='*', s=100, color='yellow')
        ax.plot(self.path[:, 0], self.path[:, 1], '--gs', lw=6, marker='*', color='green')
        # ax1.plot(self.path[:, 0], self.path[:, 1], 'k-', lw=10)
        plt.show(block=False)

        self.tf_writer.add_image("maps/world_map", self._convert_plot_to_image(figure=fig),
                                 global_step=self.global_step, dataformats='HWC')


        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 50))
        ax.set_title('observed_world_map')
        sns.heatmap(self.observed_world_map[0, :, :], ax=ax)
        ax.plot(self.path[:, 0], self.path[:, 1], '--gs', lw=6, marker='*', color='green')
        plt.show(block=False)
        self.tf_writer.add_image("maps/observed_world_map", self._convert_plot_to_image(figure=fig),
                                 global_step=self.global_step, dataformats='HWC')


        fig = plt.figure()
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
        ax.set_title('raw locations history')
        ax.plot(self.location_history[:, 0], self.location_history[:, 1], '--gs', lw=2, marker='*', color='green')
        for i, txt in enumerate(self.location_history[:, 0]):
            ax.annotate(i, (self.location_history[:, 0][i], self.location_history[:, 1][i]))
        plt.show(block=False)
        self.tf_writer.add_image("trajectories/raw_locations", self._convert_plot_to_image(figure=fig),
                                 global_step=self.global_step, dataformats='HWC')


        fig = plt.figure()
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
        ax.set_title('processed locations history')
        ax.plot(self.path[:, 0], self.path[:, 1], '--gs', lw=2, marker='*', color='green')
        for i, txt in enumerate(self.path[:, 0]):
            ax.annotate(i, (self.path[:, 0][i], self.path[:, 1][i]))
        plt.show(block=False)
        self.tf_writer.add_image("trajectories/processed_locations", self._convert_plot_to_image(figure=fig),
                                 global_step=self.global_step, dataformats='HWC')

        fig = plt.figure()
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
        ax.set_title('actions histogram')

        ax.hist(self.actions_history[:, 0:2], bins=np.arange(self.low[0], self.high[0] + 1))
        plt.show(block=False)
        self.tf_writer.add_image("histograms/actions_histogram", self._convert_plot_to_image(figure=fig),
                                 global_step=self.global_step, dataformats='HWC')

        fig = plt.figure()
        ax = plt.subplot2grid(shape=(1, 1), loc=(0, 0))
        ax.set_title('rotational actions histogram')

        ax.hist(self.actions_history[:, 2], bins=np.arange(round(self.low[-1]), round(self.high[-1] + 1)))
        plt.show(block=False)
        self.tf_writer.add_image("histograms/rotational_actions_histogram", self._convert_plot_to_image(figure=fig),
                                 global_step=self.global_step, dataformats='HWC')

        plt.close(fig='all')
        return

    def get_world_map_topic_subscriber(self, msg):
        self.world_map = np.array(msg.data).reshape([msg.layout.dim[0].size,
                                                    msg.layout.dim[1].size,
                                                    msg.layout.dim[2].size])
        self.world_map = self.world_map[int(self.current_location[-1] / 0.2)] #HARDCODED FOR NOW Assumes Z starts at 0
        mask = self.world_map.copy()
        mask[mask != -1000] = 1
        mask[mask == -1000] = 0
        self.world_map[self.world_map == -1000] = 0
        self.world_map = np.stack([self.world_map, mask], axis=0)

    def get_observed_world_map_topic_subscriber(self, msg):
        self.observed_world_map = np.array(msg.data).reshape([msg.layout.dim[0].size,
                                                    msg.layout.dim[1].size,
                                                    msg.layout.dim[2].size])
        self.observed_world_map = self.observed_world_map[int(self.current_location[-1] / 0.2)] #HARDCODED FOR NOW Assumes Z starts at 0
        mask = self.observed_world_map.copy()
        mask[mask != -1000] = 1
        mask[mask == -1000] = 0
        self.observed_world_map[self.observed_world_map == -1000] = 0
        self.observed_world_map = np.stack([self.observed_world_map, mask], axis=0)

    def render(self, mode='human'):
        return

    def set_gpu(self):
        self.set_gpu_on = True

    def reset_gpu(self):
        self.set_gpu_on = False

    def set_observations_encoder(self, model):
        self.observations_encoder = model
        self.observation_space = spaces.Box(-1000, 1000, shape=[256], dtype=np.float32)

    def reset_observations_encoder(self, model):
        self.observations_encoder = None
        self.observation_space = spaces.Box(-1000, 1000, shape=[self.esdf_map_length, self.esdf_map_length, 1],
                                            dtype=np.float32)

    def set_logging(self, value):
        if value:
            roslaunch.configure_logging(self.uuid)

    def step(self, action):
        action = [action[0]*self.action_scaling_factors[0],
                  action[1]*self.action_scaling_factors[1],
                  action[2]*self.action_scaling_factors[2]]
        print("Action is : ", action)
        if not self.validation_mode:
            self.actions_history.append(action)
        if np.abs(action[0]) < 1e-5:
            action[0] = 1e-5
        if np.abs(action[1]) < 1e-5:
            action[1] = 1e-5
        self._take_action(action)
        self.current_step += 1
        self.global_step += 1
        ob = self._get_obs()
        reward = self._calculate_reward()
        print("reward ", reward)
        done = self._is_done()
        return ob, reward, done, {}

    def set_overfit_to_one_map(self):
        self.overfit_to_one_map = True

    def set_validation_mode(self):
        self.validation_mode = True

    def set_training_mode(self):
        self.validation_mode = False

    def reset(self):
        print("RESET MAP")
        print("Episode number {} ".format(self.episode_num))
        if not self.validation_mode:
            self.episode_num += 1
        self.current_step = 0
        if not self.overfit_to_one_map:
            self.do_reset()
        else:
            self.do_reset_state()
        self.do_reset_observed_map()
        self.get_world_map()
        self.robot.reset()
        self._take_action(self.robot.default_action)
        print("FINISHED RESET")
        return self._get_obs()

    def reset_state(self):
        print("RESET STATE")
        print("Episode number {} ".format(self.episode_num))
        self.current_step = 0
        self.do_reset_state()
        self.robot.reset()
        self._take_action(self.robot.default_action)
        print("FINISHED RESET")
        return self._get_obs()

    def _take_action(self, action):
        #TODO: double check if the action of type Transform
        #wrap the action into an actual message
        translational = [*action[:2], 0]
        euler = [action[2], 0, np.pi]
        transformed_action = Transform(Vector3(*translational), Quaternion(*euler2quat(*euler)))
        resp = self.do_simulation(transformed_action)
        self.observations['collsion'] = resp.has_collision.data
        self.observations['submap'] = np.array(resp.submap.data).reshape([resp.submap.layout.dim[0].size,
                                                                          resp.submap.layout.dim[1].size, 1])
        self.observations['newly_observed_voxels'] = resp.newly_observed_voxels.data
        self.current_location = [resp.current_position.x, resp.current_position.y, resp.current_position.z]
        self.location_history.append(self.current_location)
        self.observations['executed_translation'] = [resp.T_C_old_C_end.translation.x, resp.T_C_old_C_end.translation.y]
        self.observations['executed_rotation'] = quat2euler([resp.T_C_old_C_end.rotation.w,
                                                             resp.T_C_old_C_end.rotation.x,
                                                             resp.T_C_old_C_end.rotation.y,
                                                            resp.T_C_old_C_end.rotation.z])
        self.observations['executed_action'] = [*self.observations['executed_translation'], *self.observations['executed_rotation']]
        if not self.validation_mode:
            self.executed_actions_history.append(self.observations['executed_action'])
            self.path.append([int((self.current_location[0] + 20) / 0.2), int((self.current_location[1] + 50) / 0.2)]) #HARDCODED FOR NOW
        self.robot.set_observations(self.observations)
        return

    def _get_obs(self):
        #prepare the map to 128x128
        if self.observations_encoder is not None:
            #prepare data for encoder
            inputs = self._set_unknown_spaces(self._prepare_esdf_map_shape(self.observations['submap'][:, :]))
            with torch.no_grad():
                if self.set_gpu_on:
                    inputs = Variable(torch.from_numpy(inputs)).cuda()
                else:
                    inputs = Variable(torch.from_numpy(inputs))
                inputs = inputs.float()
                result = self.observations_encoder.encode(inputs)
            return result.cpu().data.numpy()
        else:
            observations = np.array(self._prepare_esdf_map_shape(self.observations['submap'][:, :]))
            # observations = np.expand_dims(observations, axis=2)
            print("Observed Voxels SO FAR is ", observations.shape)
            return observations

    def _prepare_esdf_map_shape(self, esdf_map):
        #check for height
        if esdf_map.shape[0] > self.esdf_map_length:
            result_map = esdf_map[:self.esdf_map_length, :]
        elif esdf_map.shape[0] < self.esdf_map_length:
            missing_values = np.zeros(shape=[self.esdf_map_length - esdf_map.shape[0],
                                             esdf_map.shape[1]], dtype=np.float64)
            result_map = np.append(esdf_map, missing_values, axis=0)
        else:
            result_map = esdf_map

        # check for width
        if result_map.shape[1] > self.esdf_map_length:
            result_map = result_map[:, :self.esdf_map_length]
        elif result_map.shape[1] < self.esdf_map_length:
            missing_values = np.zeros(shape=[result_map.shape[0],
                                             self.esdf_map_length - esdf_map.shape[1]], dtype=np.float64)
            result_map = np.append(result_map, missing_values, axis=1)
        else:
            result_map = result_map

        return result_map

    def _set_unknown_spaces(self, data):
        print(data.shape)
        output = np.expand_dims(data, axis=0)
        output = np.expand_dims(output, axis=0)
        unknown_space_indicator = np.ones(shape=output.shape, dtype=np.float32)
        unknown_spaces_indicies = list(zip(*np.where(data == -1000)))
        print(unknown_space_indicator.shape)
        unknown_space_indicator[0, 0, unknown_spaces_indicies] = 0
        output = np.concatenate((output, unknown_space_indicator), axis=1)
        return output

    def _calculate_reward(self):
        return (self.observations['newly_observed_voxels'] / float(128*128)) * 100

    def _is_done(self):
        # return self.robot.is_stuck()
        print("Step number {} ".format(self.current_step))
        if self.current_step >= 20:
            self.get_observed_world_map()
            if self.save_trajectory_info and self.episode_num%self.logging_iter == 0 and not self.validation_mode:
                print("Saving trajectories history information")
                time.sleep(3)
                if self.visualize_plots:
                    self._visualize_world_map()  # TODO: MOVE THIS
                np.save(os.path.join(self.trajectory_info_dir, "path"), np.array(self.env_path))
                np.save(os.path.join(self.trajectory_info_dir, "locations_history"), np.array(self.env_location_history))
                np.save(os.path.join(self.trajectory_info_dir, "actions_history"), np.array(self.env_actions_history))
                np.save(os.path.join(self.trajectory_info_dir, "executed_actions_history"),
                        np.array(self.env_executed_actions_history))
                np.save(os.path.join(self.trajectory_info_dir, "env_maps_history"),
                        np.array(self.env_maps_history))
                np.save(os.path.join(self.trajectory_info_dir, "env_observed_maps_history"),
                        np.array(self.env_observed_maps_history))
            if not self.validation_mode:
                self.env_path.append(self.path)
                self.env_location_history.append(self.location_history)
                self.env_actions_history.append(self.actions_history)
                self.env_executed_actions_history.append(self.executed_actions_history)
                self.env_maps_history.append(self.world_map)
                self.env_observed_maps_history.append(self.observed_world_map)
            self.path = []
            self.location_history = []
            self.actions_history = []
            self.executed_actions_history = []
            return True
        else:
            return False

    def move(self):
        pass

    def set_dynamics(self):
        pass

    def get_dynamics(self):
        pass

