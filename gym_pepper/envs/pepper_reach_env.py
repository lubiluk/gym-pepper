import numpy as np
import pybullet as p
from gym import spaces
from qibullet import Camera, PepperVirtual

from . import detection
from .pepper_env import PepperEnv


class PepperReachEnv(PepperEnv):
    def __init__(self,
                 gui=False,
                 sim_steps_per_action=10,
                 dense=False,
                 head_motion=True):
        self._dense = dense
        super(PepperReachEnv,
              self).__init__(gui=gui,
                             sim_steps_per_action=sim_steps_per_action,
                             head_motion=head_motion)

    def reset(self):
        obj_pos = self._reset_scene()
        self._goal = obj_pos

        return self._get_observation()

    def close(self):
        self._robot.unsubscribeCamera(self._cam)
        super(PepperReachEnv, self).close()

    def step(self, action):
        """
        Action in terms of desired joint positions.
        The last number is the speed of the movement.
        """
        self._perform_action(action)

        obs = self._get_observation()

        is_success = self._is_success()
        is_safety_violated = self._is_table_touched(
        ) or self._is_table_displaced()

        is_object_in_sight = not (obs[-1] == obs[-2] == obs[-3] == 0.0)

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated,
            "is_object_in_sight": is_object_in_sight
        }
        reward = self._compute_reward(is_success, is_safety_violated,
                                      is_object_in_sight)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _compute_reward(self, is_success, is_safety_violated,
                        is_object_in_sight):
        if is_success:
            return 1.0

        if is_safety_violated:
            return -1.0

        if self._dense:
            if is_object_in_sight:
                return -0.001
            else:
                return -0.01

        return 0.0

    def _is_success(self):
        cont = p.getContactPoints(self._robot.getRobotModel(), self._obj)

        return len(cont) > 0 and all(36 <= c[3] <= 49 for c in cont)

    def _setup_scene(self):
        super(PepperReachEnv, self)._setup_scene()

        # Setup camera
        self._cam = self._robot.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM,
                                                resolution=Camera.K_QQVGA)

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")

    def _get_observation(self):
        goal_pos = p.getBasePositionAndOrientation(
            self._obj, physicsClientId=self._client)
        joint_p = self._get_joints_states()
        cam_pos = self._robot.getLinkPosition("CameraBottom_optical_frame")
        hand_pos = self._robot.getLinkPosition("l_hand")
        # Object position relative to camera
        inv = p.invertTransform(cam_pos[0], cam_pos[1])
        rel_pos = p.multiplyTransforms(inv[0], inv[1], goal_pos[0],
                                       goal_pos[1])
        obj_rel_pos = np.array(rel_pos[0])

        img = self._robot.getCameraFrame(self._cam)

        if not detection.is_object_in_sight(img):
            obj_rel_pos = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        return np.concatenate([
            joint_p, cam_pos[0], cam_pos[1], hand_pos[0], hand_pos[1],
            obj_rel_pos
        ]).astype(np.float32)

        # return np.concatenate([joint_p, obj_rel_pos]).astype(np.float32)
