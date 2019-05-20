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
import numpy as np
import gym
import gym_TSDF_explore
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.ddpg.policies import LnCnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, DDPG
from gym_TSDF_explore.configs.env_config import ENV_CONFIG
from TSDF_explore.state_encoder.encoding_state import EncodingState
from TSDF_explore.policies.policy_loader import ModelLoader
from TSDF_explore.ros_recorder.data_recorder import DataRecorder
from tensorboardX import SummaryWriter
import time
import os
import shutil
from TSDF_explore.policies.random_policy import RandomPolicy

logging_global_dir = "./logs/ppo2_policy/"


def make_env(rank, seed=0):
    def _init():
        env = gym.make('TSDF_explore-v0')
        env.seed(seed + rank)
        trajectory_info_dir = os.path.join(logging_global_dir + 'environment_logging_' + str(rank))
        if os.path.exists(trajectory_info_dir):
            shutil.rmtree(trajectory_info_dir)
        tf_writer = SummaryWriter(log_dir=trajectory_info_dir)
        env.initialize_environment(name_space=rank, tf_writer=tf_writer, trajectory_info_dir=trajectory_info_dir,
                                   save_trajectory_info=True, visualize_plots=True)
        env.set_training_mode()
        # env.set_overfit_to_one_map()
        return env
    set_global_seeds(seed)
    return _init


def evaluate_episode(env, model, num_steps=None):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_reward = 0.0
    time_step = 0
    obs = env.reset_state()
    while True:
        print("time step number {}".format(time_step))
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)
        episode_reward += rewards
        time_step += 1
        if num_steps is None:
            if dones:
                print("DONE")
                break
        else:
            if time_step >= num_steps:
                break
    return episode_reward


def train_exploring_policy_parallel():

    num_cpu = 4
    validate_every_timesteps = 256 * num_cpu
    total_time_steps = 100000
    # spawn a new environment
    env = SubprocVecEnv([make_env(rank=i) for i in range(num_cpu)])
    model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=logging_global_dir)
    testing_env = gym.make('TSDF_explore-v0')
    trajectory_info_dir = os.path.join(logging_global_dir + 'environment_logging_validation')
    if os.path.exists(trajectory_info_dir):
        shutil.rmtree(trajectory_info_dir)
    validation_tf_writer = SummaryWriter(log_dir=trajectory_info_dir)
    testing_env.initialize_environment(name_space=num_cpu, tf_writer=validation_tf_writer,
                                       trajectory_info_dir=trajectory_info_dir,
                                       save_trajectory_info=False, visualize_plots=False)
    testing_env.set_validation_mode()

    for i in range(int(total_time_steps/validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps, tb_log_name="ppo2_non_encoded_state",
                    reset_num_timesteps=False)  # multiple of 128
        print("Saving the model")
        model.save("model_v3")
        print("Validating")
        validation_model = PPO2.load("model_v3")
        random_policy_model = RandomPolicy()
        policy_models = [validation_model, random_policy_model]
        num_episodes_to_run = 30
        episode_mean_rewards = np.zeros(shape=[num_episodes_to_run, 2], dtype=np.float64)
        for j in range(num_episodes_to_run):
            obs = testing_env.reset()
            print("EPISODE NUMBER ________________________________", i)
            for k in range(2):
                episode_reward = evaluate_episode(testing_env, policy_models[k], num_steps=None)
                episode_mean_rewards[j, k] = episode_reward  # sum for now
        #now plot the shit
        validation_tf_writer.add_scalar('data/validation_reward_random_policy',
                                        np.mean(episode_mean_rewards[:, 1]), (i+1) * validate_every_timesteps)
        validation_tf_writer.add_scalar('data/validation_reward_trained_policy',
                                        np.mean(episode_mean_rewards[:, 0]), (i+1) * validate_every_timesteps)
        time.sleep(5)
    return
