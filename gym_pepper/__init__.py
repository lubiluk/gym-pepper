from gym.envs.registration import register

register(
    id='PepperPush-v0',
    entry_point='gym_pepper.envs:PepperPushEnv',
)

register(
    id='PepperArmVelPush-v0',
    entry_point='gym_pepper.envs:PepperArmVelPushEnv',
)

register(
    id='PepperArmPosPush-v0',
    entry_point='gym_pepper.envs:PepperArmPosPushEnv',
)