# Some bits are based on:
# https://github.com/softbankrobotics-research/qi_gym/blob/master/envs/throwing_env.py

import gym
import os.path
import numpy as np
import pybullet as p
from .pepper_reach_env import PepperReachEnv
from gym import spaces
from qibullet import PepperVirtual, Camera


class PepperReachCamEnv(PepperReachEnv):
    def close(self):
        self._robot.unsubscribeCamera(self._cam)
        super(PepperReachCamEnv, self).close()

    def step(self, action):
        """
        Action in terms of desired joint positions. Last number is the speed of the movement.
        """
        self._perform_action(action)

        obs = self._get_observation()

        is_success = self._is_success()
        is_safety_violated = self._is_table_touched(
        ) or self._is_table_displaced()
        obj_pos = self._get_object_pos()

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated,
            "object_position": obj_pos
        }
        reward = self._compute_reward(is_success, is_safety_violated)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _setup_scene(self):
        super(PepperReachCamEnv, self)._setup_scene()

        # Setup camera
        self._cam = self._robot.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM,
                                                resolution=Camera.K_QQVGA)

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
                joints_state=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=obs["joints_state"].shape,
                    dtype=obs["joints_state"].dtype,
                ),
            ))

    def _get_observation(self):
        img = self._robot.getCameraFrame(self._cam)

        joint_p = self._robot.getAnglesPosition(self.CONTROLLABLE_JOINTS)
        # joint velocities are not available on real Pepper
        # joint_v = self._robot.getAnglesVelocity(CONTROLLABLE_JOINTS)

        result = {
            "camera": img,
            "joints_state": np.array(joint_p, dtype=np.float32)
        }

        return result

    def _get_object_pos(self):
        goal_pos = self._goal
        cam_idx = self._robot.link_dict["CameraBottom_optical_frame"].getIndex(
        )
        cam_pos = p.getLinkState(self._robot.getRobotModel(),
                                 cam_idx,
                                 physicsClientId=self._client)[0]
        # Object position relative to camera
        return np.array(goal_pos) - np.array(cam_pos)
