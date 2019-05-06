import gym
import gym_TSDF_explore
from TSDF_explore.policies.policy_loader import ModelLoader
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np
from TSDF_explore.policies.random_policy import RandomPolicy


class BenchMark(object):
    def __init__(self):
        self.env = gym.make('TSDF_explore-v0')
        self.vectorized_env = None
        self.policy_models = []
        self.time_steps = []

    def add_policy(self, model, time_steps):
        self.policy_models.append(model)
        self.time_steps.append(time_steps)
        return

    def add_mlp_encoder_policy(self):
        model_loader = ModelLoader(model_class="state_encoder_v1",
                                   trained_model_path="pretrained_models/state_autoencoder_v4.pth", gpu=True)
        self.env.set_observations_encoder(model_loader.get_inference_model())
        self.env.set_gpu()
        self.vectorized_env = DummyVecEnv([lambda: self.env])

        model = PPO2(MlpPolicy, self.vectorized_env, verbose=1, tensorboard_log="./logs/")
        model.load("model_v1_encoded_state")
        self.add_policy(model, None)

    def add_mlp_non_encoder_policy(self):
        self.vectorized_env = DummyVecEnv([lambda: self.env])

        model = PPO2(MlpPolicy, self.vectorized_env, verbose=1, tensorboard_log="./logs/")
        model.load("model_v1_non_encoded_state")
        self.add_policy(model, None)
        self.add_policy(model, None)

    def add_random_policy(self):
        self.add_policy(RandomPolicy(), None)


    # def test_policy(self):
    #     model_loader = ModelLoader(model_class="state_encoder_v1",
    #                                trained_model_path="pretrained_models/state_autoencoder_v4.pth", gpu=True)
    #     self.env.set_observations_encoder(model_loader.get_inference_model())
    #     self.env.set_gpu()
    #     env = DummyVecEnv([lambda: self.env])
    #
    #     model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
    #     model.load("model_v1_encoded_state")
    #     print("Finished training")
    #     print("Saving the model")
    #     model.save("model_v1")
    #     mean_100ep_reward = self.evaluate(env, model)
    #     return mean_100ep_reward


    def evaluate(self, num_episodes=2):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward for the last 100 episodes
        """
        episode_mean_rewards = np.zeros(shape=[num_episodes, len(self.policy_models)], dtype=np.float64)
        for i in range(num_episodes):
            obs = self.vectorized_env.reset()
            print("EPISODE NUMBER ________________________________", i)
            for j in range(len(self.policy_models)):
                episode_reward = self.evaluate_episode(self.policy_models[j], num_steps=self.time_steps[j])
                episode_mean_rewards[i, j] = episode_reward #sum for now
        return episode_mean_rewards

    def evaluate_episode(self, model, num_steps=None):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward for the last 100 episodes
        """
        episode_reward = 0.0
        time_step = 0
        obs = self.vectorized_env.env_method('reset_state')
        while True:
            print("time step number {}".format(time_step))
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, rewards, dones, info = self.vectorized_env.step(action, eval=True)
            episode_reward += rewards[0]
            time_step += 1
            if num_steps is None:
                if dones[0]:
                    print ("DONE")
                    break
            else:
                if time_step >= num_steps:
                    break
        return episode_reward
