# Some bits are based on:
# https://github.com/softbankrobotics-research/qi_gym/blob/master/envs/throwing_env.py

import gym
import os.path
import numpy as np
import pybullet as p
from .pepper_reach_env import PepperReachEnv
from gym import spaces
from . import detection


class PepperReachCamEnv(PepperReachEnv):
    def step(self, action):
        """
        Action in terms of desired joint positions. Last number is the speed of the movement.
        """
        self._perform_action(action)

        obs = self._get_observation()

        is_success = self._is_success()
        is_safety_violated = self._is_table_touched(
        ) or self._is_table_displaced()
        obj_pos = self._get_object_pos(obs["camera"])
        is_object_in_sight = is_object_in_sight = not (obj_pos[-1] == obj_pos[-2] == obj_pos[-3] == 0.0)

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated,
            "object_position": obj_pos,
            "is_object_in_sight": is_object_in_sight
        }
        reward = self._compute_reward(is_success, is_safety_violated,
                                      is_object_in_sight)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Dict(
            dict(
                camera=spaces.Box(
                    0,
                    255,
                    shape=obs["camera"].shape,
                    dtype=obs["camera"].dtype,
                ),
                camera_pose=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["camera_pose"].shape,
                    dtype=obs["camera_pose"].dtype,
                ),
                joints_state=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["joints_state"].shape,
                    dtype=obs["joints_state"].dtype,
                ),
            ))

    def _get_observation(self):
        img = self._robot.getCameraFrame(self._cam)

        joint_p = self._get_joints_states()
        cam_pos = self._robot.getLinkPosition("CameraBottom_optical_frame")

        result = {
            "camera":
            img,
            "camera_pose":
            np.concatenate([cam_pos[0], cam_pos[1]]).astype(np.float32),
            "joints_state":
            np.array(joint_p, dtype=np.float32)
        }

        return result

    def _get_object_pos(self, img):
        goal_pos = p.getBasePositionAndOrientation(
            self._obj, physicsClientId=self._client)
        cam_pos = self._robot.getLinkPosition("CameraBottom_optical_frame")
        # Object position relative to camera
        inv = p.invertTransform(cam_pos[0], cam_pos[1])
        rel_pos = p.multiplyTransforms(inv[0], inv[1], goal_pos[0],
                                       goal_pos[1])

        if not detection.is_object_in_sight(img):
            return np.array((0.0, 0.0, 0.0), dtype=np.float32)

        return np.array(rel_pos[0])
