import gym
import gym_pepper

env = gym.make('PepperPush-v0')

env.reset()

for _ in range(1000):
    env.step([1.0] * 11)

env.reset()

for _ in range(1000):
    env.step([0.0] * 11)