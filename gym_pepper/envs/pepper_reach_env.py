from .pepper_env import PepperEnv
import pybullet as p
import numpy as np
from gym import spaces


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
        is_looking_at_table = obs[-3:].any()

        info = {
            "is_success": is_success,
            "is_safety_violated": is_safety_violated
        }
        reward = self._compute_reward(is_success, is_safety_violated, is_looking_at_table)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def _compute_reward(self, is_success, is_safety_violated, is_looking_at_table):
        if is_success:
            return 1.0

        if is_safety_violated:
            return -1.0

        if self._dense:
            if is_looking_at_table:
                return -0.001
            else:
                return -0.01

        return 0.0

    def _is_success(self):
        cont = p.getContactPoints(self._robot.getRobotModel(), self._obj)

        return len(cont) > 0 and all(36 <= c[3] <= 49 for c in cont)

    def _get_observation_space(self):
        obs = self._get_observation()

        return spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype="float32")

    def _get_observation(self):
        goal_pos = self._goal
        joint_p = self._robot.getAnglesPosition(self.CONTROLLABLE_JOINTS)
        # joint velocities are not available on real Pepper
        # joint_v = self._robot.getAnglesVelocity(CONTROLLABLE_JOINTS)
        cam_pos = self._robot.getLinkPosition("CameraBottom_optical_frame")
        # Object position relative to camera
        obj_rel_pos = np.array(goal_pos) - np.array(cam_pos[0])

        v, _ = p.multiplyTransforms(cam_pos[0], cam_pos[1], (0, 0, 1),
                                    (0, 0, 0, 1))
        hit_id = p.rayTest(cam_pos[0], v)[0][0]

        if (hit_id != self._table) and (hit_id != self._obj):
            obj_rel_pos = np.array((0.0, 0.0, 0.0), dtype=np.float32)

        return np.concatenate([joint_p, cam_pos[0], cam_pos[1],
                               obj_rel_pos]).astype(np.float32)
