import gym
import gym_pepper
import time
import numpy as np
import cv2

start = time.time()
env = gym.make("PepperPushCam-v0", gui=False)
end = time.time()
print("=== Make === {}".format(end - start))

start = time.time()
env.reset()
end = time.time()
print("=== Reset === {}".format(end - start))

# start = time.time()
# for _ in range(100):
#     env.step([1.0] * 11)
# for _ in range(100):
#     env.step([-1.0] * 11)
# end = time.time()
# print("=== Act1 === {}".format(end - start))

# start = time.time()
# env.reset()
# end = time.time()
# print("=== Reset === {}".format(end - start))

start = time.time()
for _ in range(1000):
    angles = np.random.sample(9) * 2 - 1
    speed = np.random.random_sample()
    action = np.concatenate([angles, [speed]])
    o, r, d, i = env.step(action)
    cv2.imshow("synthetic bottom camera", o["observation"]["camera_bottom"])
    cv2.waitKey(1)

    if d:
        env.reset()

    if r == 1.0:
        print("Touch!")

end = time.time()
print("=== Act2 === {}".format(end - start))

env.close()
