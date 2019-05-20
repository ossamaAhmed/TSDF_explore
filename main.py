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
from TSDF_explore.state_encoder.encoding_state import EncodingState
from TSDF_explore.policies.policy_loader import ModelLoader
from TSDF_explore.ros_recorder.data_recorder import DataRecorder
from tensorboardX import SummaryWriter
import os
import shutil
from TSDF_explore.benchmark.deploy import BenchMark
import time
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from TSDF_explore.train.train_ppo2 import train_exploring_policy_parallel


def train_encoder():
    state_model = EncodingState(prepare_data_from_sim=True)
    state_model.train(num_epochs=500, model_path="pretrained_models/state_autoencoder_v4.pth", gpu=True)


def test_encoder():
    state_model = EncodingState(prepare_data_from_sim=True)
    state_model.test(model_path="pretrained_models/state_autoencoder_v4.pth", gpu=True)


def train_exploring_policy():
    env = gym.make('TSDF_explore-v0')
    model_loader = ModelLoader(model_class="state_encoder_v1",
                               trained_model_path="pretrained_models/state_autoencoder_v4.pth", gpu=True)
    # env.set_observations_encoder(model_loader.get_inference_model())
    # env.set_gpu()

    env.initialize_environment(name_space=1)
    # env.set_overfit_to_one_map()
    env = DummyVecEnv([lambda: env])
    # the noise objects for DDPG
    # n_actions = env.action_space.shape[-1]
    # param_noise = None
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    #
    # model = DDPG(LnCnnPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

    model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=100) #multuple of 128
    print("Finished training")
    print("Saving the model")
    model.save("model_v2_non_encoded_state_CNN")
    return model, env


def record_random_dataset():
    dr = DataRecorder((129, 129), 2000, 100)
    dr.record_data()


def main():
    # record_random_dataset()
    # benchmark = BenchMark()
    # benchmark.add_mlp_non_encoder_policy()
    # benchmark.add_mlp_non_encoder_policy()
    # benchmark.add_random_policy()
    # print("EVALUATED THE POLICY: {}".format(benchmark.evaluate(num_episodes=500)))
    # test_encoder()
    train_exploring_policy_parallel()
    # print("EVALUATED THE POLICY: {}".format(test_exploring_policy()))
    # train_encoder()
    #ERROR: 0413 17:27:23.913967 10304 simulator.cc:487] Moving from -6.99358 0019.899 03.38787 to -7.11874 019.3573 003.1897...
# I0413 17:27:23.914000 10304 simulator.cc:324] Shortest rotation between 1 0 0 and 000.999999 0.00127657 0000000000 is 0.0740187deg around 0 0 1
# F0413 17:27:23.914049 10304 simulator.cc:337] Check failed: std::abs(Ci_p_Co_Cn.y()) < kEpsilon (1.52978e-05 vs. 1e-05)


if __name__ == "__main__":
    main()
