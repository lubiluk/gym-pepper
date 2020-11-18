import gym
import gym_pepper
import time
import numpy as np

start = time.time()
env = gym.make('PepperPush-v0', gui=False)
end = time.time()
print("=== Make === {}".format(end - start))

start = time.time()
env.reset()
end = time.time()
print("=== Reset === {}".format(end - start))

start = time.time()
for _ in range(15000):
    env.step([1.0] * 11)
end = time.time()
print("=== Act1 === {}".format(end - start))

start = time.time()
env.reset()
end = time.time()
print("=== Reset === {}".format(end - start))

start = time.time()
for _ in range(15000):
    env.step(np.random.sample(11))
end = time.time()
print("=== Act2 === {}".format(end - start))
