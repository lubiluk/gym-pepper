# Some bits are based on:
# https://github.com/softbankrobotics-research/qi_gym/blob/master/envs/throwing_env.py

import gym
import numpy as np
import pybullet as p
import os.path
from gym import spaces
from .pepper_env import PepperEnv, CONTROLLABLE_JOINTS

DISTANCE_THRESHOLD = 0.04


class PepperPushEnv(PepperEnv):
    def __init__(self, gui=False, sim_steps_per_action=10):
        super(PepperPushEnv,
              self).__init__(gui=gui,
                             sim_steps_per_action=sim_steps_per_action)

    def reset(self):
        self._reset_scene()
        self._goal = self._sample_goal()

        if self._gui:
            self._place_ghosts()

        return self._get_observation()

    def step(self, action):
        self._perform_action(action)

        obs = self._get_observation()

        is_success = self._is_success(obs["achieved_goal"], self._goal)
        is_safety_violated = self._is_table_touched(
        ) or self._is_table_displaced()

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated
        }
        reward = self.compute_reward(obs["achieved_goal"], self._goal, info)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return (np.linalg.norm(desired_goal - achieved_goal, axis=-1) <
                DISTANCE_THRESHOLD).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        return (np.linalg.norm(desired_goal - achieved_goal, axis=-1) <
                DISTANCE_THRESHOLD)

    def _setup_scene(self):
        super(PepperPushEnv, self)._setup_scene()

        if self._gui:
            # load ghosts
            self._ghost = p.loadURDF(
                "brick/brick_ghost.urdf",
                self._obj_start_pos,
                self._obj_init_ori,
                physicsClientId=self._client,
                useFixedBase=True,
            )

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Dict(
            dict(
                desired_goal=spaces.Box(-np.inf,
                                        np.inf,
                                        shape=obs["achieved_goal"].shape,
                                        dtype="float32"),
                achieved_goal=spaces.Box(-np.inf,
                                         np.inf,
                                         shape=obs["achieved_goal"].shape,
                                         dtype="float32"),
                observation=spaces.Box(-np.inf,
                                       np.inf,
                                       shape=obs["observation"].shape,
                                       dtype="float32"),
            ))

    def _get_observation(self):
        obj_pos = p.getBasePositionAndOrientation(
            self._obj, physicsClientId=self._client)[0]
        obj_vel = p.getBaseVelocity(self._obj, physicsClientId=self._client)[0]
        joint_p = self._robot.getAnglesPosition(CONTROLLABLE_JOINTS)
        # joint velocities are not available on real Pepper
        # joint_v = self._robot.getAnglesVelocity(CONTROLLABLE_JOINTS)
        cam_pos = self._robot.getLinkPosition("CameraBottom_optical_frame")
        # Object position relative to camera
        obj_rel_pos = np.array(obj_pos) - np.array(cam_pos[0])

        v, _ = p.multiplyTransforms(cam_pos[0], cam_pos[1], (0, 0, 1),
                                    (0, 0, 0, 1))
        hit_id = p.rayTest(cam_pos[0], v)[0][0]

        if hit_id != self._table:
            obj_rel_pos = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        return {
            "observation":
            np.concatenate([
                obj_pos, obj_vel, joint_p, cam_pos[0], cam_pos[1], obj_rel_pos
            ]).astype(np.float32),
            "achieved_goal":
            np.array(obj_pos, dtype=np.float32),
            "desired_goal":
            self._goal,
        }

    def _place_ghosts(self):
        p.resetBasePositionAndOrientation(
            self._ghost,
            posObj=self._goal,
            ornObj=self._obj_init_ori,
            physicsClientId=self._client,
        )
