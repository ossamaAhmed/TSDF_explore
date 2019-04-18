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
import time


def train_encoder():
    state_model = EncodingState(do_randomize_unknown_spaces=True)
    state_model.train(num_epochs=2, model_path="pretrained_models/state_autoencoder_v3.pth", gpu=True)


def test_encoder():
    state_model = EncodingState(do_randomize_unknown_spaces=False)
    state_model.test(model_path="pretrained_models/state_autoencoder_v1.pth", gpu=False)


def train_exploring_policy():
    env = gym.make('TSDF_explore-v0')
    model_loader = ModelLoader(model_class="state_encoder_v1",
                               trained_model_path="pretrained_models/state_autoencoder_v3.pth", gpu=False)
    env.set_observations_encoder(model_loader.get_inference_model())
    # env.set_gpu()
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=100000)
    print("Finished training")
    return model, env


def test_exploring_policy(env, model):
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("DIED")
            time.sleep(20)


def main():
    # test_encoder()
    train_exploring_policy()
    # train_encoder()
    #ERROR: 0413 17:27:23.913967 10304 simulator.cc:487] Moving from -6.99358 0019.899 03.38787 to -7.11874 019.3573 003.1897...
# I0413 17:27:23.914000 10304 simulator.cc:324] Shortest rotation between 1 0 0 and 000.999999 0.00127657 0000000000 is 0.0740187deg around 0 0 1
# F0413 17:27:23.914049 10304 simulator.cc:337] Check failed: std::abs(Ci_p_Co_Cn.y()) < kEpsilon (1.52978e-05 vs. 1e-05)


if __name__ == "__main__":
    main()