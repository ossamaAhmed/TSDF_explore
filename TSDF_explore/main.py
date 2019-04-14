import rospy
from voxblox_rl_simulator.srv import *
from geometry_msgs.msg import Vector3, Quaternion, Transform, TransformStamped
import numpy as np
import gym
import gym_TSDF_explore
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time


def main():
    env = gym.make('TSDF_explore-v0')
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=1000)

    # obs = env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         print("DIED")
    #         time.sleep(20)

    # current_observations = env.reset()
    # print(current_observations)
    # print(np.min(current_observations[1]))
    # print(np.max(current_observations[1]))
    # for i in range(100000):
    #     rand_sample = np.random.randint(0,20, size=[3])
    #     rand_sample = np.append(rand_sample, [0, 0, 0])
    #     ob, reward, done, _ = env.step(rand_sample)
    #     print(np.min(ob[1]))
    #     print(np.max(ob[1]))
    #     print(ob)
    #     print(reward)
    #     print(done)
    # env.close()
    # state_model = encodingState()
    # state_model.train()
    # dataset = SDF()


if __name__ == "__main__":
    main()