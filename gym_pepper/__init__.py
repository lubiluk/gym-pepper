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
    id='PepperPushCam-v0',
    entry_point='gym_pepper.envs:PepperPushCamEnv',
)