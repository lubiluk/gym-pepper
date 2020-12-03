import gym
import gym_pepper
import time
import numpy as np

start = time.time()
env = gym.make('PepperReach-v0', gui=True)
end = time.time()
print("=== Make === {}".format(end - start))

start = time.time()
env.reset()
end = time.time()
print("=== Reset === {}".format(end - start))

start = time.time()
for _ in range(100):
    env.step([1.0] * 11)
for _ in range(100):
    env.step([-1.0] * 11)
end = time.time()
print("=== Act1 === {}".format(end - start))

start = time.time()
env.reset()
end = time.time()
print("=== Reset === {}".format(end - start))

start = time.time()
for _ in range(100):
    env.step(np.random.sample(11) * 2 - 1)
end = time.time()
print("=== Act2 === {}".format(end - start))
