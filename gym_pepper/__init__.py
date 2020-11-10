from gym.envs.registration import register

register(
    id='PepperPush-v0',
    entry_point='gym_pepper.envs:PepperPushEnv',
)