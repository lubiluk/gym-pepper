import numpy as np
import pybullet as p
from gym import spaces
from qibullet import Camera, PepperVirtual

from . import detection
from .pepper_env import PepperEnv


class PepperReachDepthEnv(PepperEnv):
    def __init__(self,
                 gui=False,
                 sim_steps_per_action=10,
                 dense=False,
                 head_motion=True):
        self._dense = dense
        super(PepperReachDepthEnv,
              self).__init__(gui=gui,
                             sim_steps_per_action=sim_steps_per_action,
                             head_motion=head_motion)

    def reset(self):
        obj_pos = self._reset_scene()
        self._goal = obj_pos

        return self._get_observation()

    def close(self):
        self._robot.unsubscribeCamera(self._depth)
        super(PepperReachDepthEnv, self).close()

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
        obj_pos = self._get_object_pos()

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated,
            "object_position": obj_pos
        }
        reward = self._compute_reward(is_success, is_safety_violated)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _compute_reward(self, is_success, is_safety_violated):
        if is_success:
            return 1.0

        if is_safety_violated:
            return -1.0

        if self._dense:
            return -0.01

        return 0.0

    def _is_success(self):
        cont = p.getContactPoints(self._robot.getRobotModel(), self._obj)

        return len(cont) > 0 and all(36 <= c[3] <= 49 for c in cont)

    def _setup_scene(self):
        super(PepperReachDepthEnv, self)._setup_scene()

        # Setup camera
        self._depth = self._robot.subscribeCamera(
            PepperVirtual.ID_CAMERA_DEPTH, resolution=Camera.K_QQVGA)

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Dict(
            dict(
                camera=spaces.Box(
                    0,
                    65535,
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
        img = self._robot.getCameraFrame(self._depth)

        joint_p = self._get_joints_states()
        cam_pos = self._robot.getLinkPosition("CameraDepth_optical_frame")

        result = {
            "camera":
            img,
            "camera_pose":
            np.concatenate([cam_pos[0], cam_pos[1]]).astype(np.float32),
            "joints_state":
            np.array(joint_p, dtype=np.float32)
        }

        return result

    def _get_object_pos(self):
        goal_pos = p.getBasePositionAndOrientation(
            self._obj, physicsClientId=self._client)
        cam_idx = self._robot.link_dict["CameraDepth_optical_frame"].getIndex(
        )
        cam_pos = p.getLinkState(self._robot.getRobotModel(),
                                 cam_idx,
                                 physicsClientId=self._client)
        # Object position relative to camera
        inv = p.invertTransform(cam_pos[0], cam_pos[1])
        rel_pos = p.multiplyTransforms(inv[0], inv[1], goal_pos[0],
                                       goal_pos[1])

        return np.array(rel_pos[0])
