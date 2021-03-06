from gym.envs.registration import register

register(
    id='PepperPush-v0',
    entry_point='gym_pepper.envs:PepperPushEnv',
)

register(
    id='PepperReach-v0',
    entry_point='gym_pepper.envs:PepperReachEnv',
)

register(
    id='PepperReachCam-v0',
    entry_point='gym_pepper.envs:PepperReachCamEnv',
)

register(
    id='PepperPushCam-v0',
    entry_point='gym_pepper.envs:PepperPushCamEnv',
)

register(
    id='PepperReachDepth-v0',
    entry_point='gym_pepper.envs:PepperReachDepthEnv',
)