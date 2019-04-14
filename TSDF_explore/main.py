import numpy as np
import gym
import gym_TSDF_explore
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from TSDF_explore.state_encoder.encoding_state import encodingState
from TSDF_explore.policies.policy_loader import ModelLoader
import time


def train_encoder():
    state_model = encodingState()
    state_model.train(num_epochs=2, model_path="pretrained_models/state_autoencoder_v2.pth")


def test_encoder():
    state_model = encodingState()
    state_model.test(model_path="pretrained_models/state_autoencoder_v2.pth")


def train_exploring_policy():
    env = gym.make('TSDF_explore-v0')
    model_loader = ModelLoader(model_class="state_encoder_v1",
                               trained_model_path="pretrained_models/state_autoencoder_v2.pth")
    env.set_observations_encoder(model_loader.get_inference_model())
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
    test_encoder()
    # train_exploring_policy()


if __name__ == "__main__":
    main()