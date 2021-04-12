# Some bits are based on:
# https://github.com/softbankrobotics-research/qi_gym/blob/master/envs/throwing_env.py

import gym
import numpy as np
import pybullet as p
import os.path
from gym import spaces
from .pepper_env import PepperEnv
from . import detection
from qibullet import Camera, PepperVirtual

DISTANCE_THRESHOLD = 0.04


class PepperPushEnv(PepperEnv, gym.GoalEnv):
    def __init__(self, gui=False, sim_steps_per_action=10, head_motion=True):
        super(PepperPushEnv,
              self).__init__(gui=gui,
                             sim_steps_per_action=sim_steps_per_action,
                             head_motion=head_motion)

    def reset(self):
        self._reset_scene()
        self._goal = self._sample_goal()

        if self._gui:
            self._place_ghosts()

        return self._get_observation()

    def close(self):
        if self._robot:
            self._robot.unsubscribeCamera(self._cam)
        super(PepperPushEnv, self).close()

    def step(self, action):
        self._perform_action(action)

        obs = self._get_observation()

        is_success = self._is_success(obs["achieved_goal"], self._goal[:2])
        is_safety_violated = self._is_table_touched(
        ) or self._is_table_displaced()

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated
        }
        reward = self.compute_reward(obs["achieved_goal"], self._goal[:2],
                                     info)
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

        # Setup camera
        self._cam = self._robot.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM,
                                                resolution=Camera.K_QQVGA)

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
        obj_pos = p.getBasePositionAndOrientation(self._obj,
                                                  physicsClientId=self._client)
        joint_p = self._get_joints_states()
        cam_pos = self._robot.getLinkPosition("CameraBottom_optical_frame")
        hand_pos = self._robot.getLinkPosition("l_hand")
        # Object position relative to camera
        inv = p.invertTransform(cam_pos[0], cam_pos[1])
        obj_rel_pos = np.array(
            p.multiplyTransforms(inv[0], inv[1], obj_pos[0], obj_pos[1])[0])
        goal_rel_pos = np.array(
            p.multiplyTransforms(inv[0], inv[1], self._goal, obj_pos[1])[0])

        img = self._robot.getCameraFrame(self._cam)

        if not detection.is_object_in_sight(img):
            obj_rel_pos = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        return {
            "observation":
            np.concatenate(
                [joint_p, cam_pos[0], cam_pos[1], obj_rel_pos,
                 goal_rel_pos]).astype(np.float32),
            "achieved_goal":
            np.array(obj_pos[0][:2], dtype=np.float32),
            "desired_goal":
            self._goal[:2],
        }

    def _place_ghosts(self):
        p.resetBasePositionAndOrientation(
            self._ghost,
            posObj=self._goal,
            ornObj=self._obj_init_ori,
            physicsClientId=self._client,
        )
