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
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from TSDF_explore.state_encoder.encoding_state import EncodingState
from TSDF_explore.policies.policy_loader import ModelLoader
from TSDF_explore.ros_recorder.data_recorder import DataRecorder
import time


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
    env.set_observations_encoder(model_loader.get_inference_model())
    env.set_gpu()
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=10000000) #multuple of 128
    print("Finished training")
    print("Saving the model")
    model.save("model_v1")
    return model, env


def evaluate(env, model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        print("time step number {}".format(i))
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0] and i < num_steps - 1:
            # obs = env.reset() #environment already did reset
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


def test_exploring_policy():
    env = gym.make('TSDF_explore-v0')
    model_loader = ModelLoader(model_class="state_encoder_v1",
                               trained_model_path="pretrained_models/state_autoencoder_v4.pth", gpu=True)
    env.set_observations_encoder(model_loader.get_inference_model())
    env.set_gpu()
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
    model.load("model_v1")
    print("Finished training")
    print("Saving the model")
    model.save("model_v1")
    mean_100ep_reward = evaluate(env, model)
    return mean_100ep_reward


def record_random_dataset():
    dr = DataRecorder((129, 129), 2000, 100)
    dr.record_data()


def main():
    # record_random_dataset()
    # test_encoder()
    train_exploring_policy()
    # print("EVALUATED THE POLICY: {}".format(test_exploring_policy()))
    # train_encoder()
    #ERROR: 0413 17:27:23.913967 10304 simulator.cc:487] Moving from -6.99358 0019.899 03.38787 to -7.11874 019.3573 003.1897...
# I0413 17:27:23.914000 10304 simulator.cc:324] Shortest rotation between 1 0 0 and 000.999999 0.00127657 0000000000 is 0.0740187deg around 0 0 1
# F0413 17:27:23.914049 10304 simulator.cc:337] Check failed: std::abs(Ci_p_Co_Cn.y()) < kEpsilon (1.52978e-05 vs. 1e-05)


if __name__ == "__main__":
    main()