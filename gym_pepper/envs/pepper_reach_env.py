# Some bits are based on:
# https://github.com/softbankrobotics-research/qi_gym/blob/master/envs/throwing_env.py

from pathlib import Path

import gym
import numpy as np
import pybullet as p
from gym import error, spaces, utils
from gym.utils import seeding
from qibullet import PepperVirtual, SimulationManager

DISTANCE_THRESHOLD = 0.04
CONTROLLABLE_JOINTS = [
    "HipRoll",
    "HeadYaw",
    "HeadPitch",
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
]


class PepperReachEnv(gym.GoalEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, gui=False, sim_steps_per_action=10, max_motion_speed=0.5, dense=False
    ):
        self._sim_steps = sim_steps_per_action
        self._max_speeds = [max_motion_speed] * len(CONTROLLABLE_JOINTS)
        self._gui = gui
        self._dense = dense

        self._setup_scene()

        self._goal = self._sample_goal()
        obs = self._get_observation()

        self.action_space = spaces.Box(
            -2.0857, 2.0857, shape=(len(CONTROLLABLE_JOINTS),), dtype="float32"
        )

        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )

    def reset(self):
        self._reset_scene()

        return self._get_observation()

    def step(self, action):
        action = list(action)
        assert len(action) == len(self.action_space.high.tolist())

        self._robot.setAngles(CONTROLLABLE_JOINTS, action, self._max_speeds)

        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_observation()

        is_success = self._is_success()
        info = {
            "is_success": is_success,
        }
        reward = self.compute_reward(info)
        done = is_success or self._is_table_displaced()

        return (obs, reward, done, info)

    def render(self, mode="human"):
        pass

    def close(self):
        self._simulation_manager.stopSimulation(self._client)

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def _is_success(self):
        cont = p.getContactPoints(self._robot.getRobotModel(), self._obj)

        return len(cont) > 0

    def compute_reward(self, info):
        if info["is_success"]:
            return 1.0

        if self._dense:
            hand_idx = self._robot.link_dict["l_hand"].getIndex()
            hand_pos = np.array(
                p.getLinkState(
                    self._robot.getRobotModel(), hand_idx, physicsClientId=self._client
                )[0]
            )
            obj_pos = np.array(
                p.getBasePositionAndOrientation(
                    self._obj, physicsClientId=self._client
                )[0]
            )
            return np.linalg.norm(obj_pos - hand_pos, axis=-1).astype(np.float32)

        return 0.0

    def _setup_scene(self):
        self._simulation_manager = SimulationManager()
        self._client = self._simulation_manager.launchSimulation(
            gui=self._gui, auto_step=False
        )

        self._robot = self._simulation_manager.spawnPepper(
            self._client, spawn_ground_plane=True
        )

        self._robot.goToPosture("Stand", 1.0)

        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)

        self._robot.setAngles(
            ["KneePitch", "HipPitch", "LShoulderPitch"], [0.33, -0.9, 0.0], [0.5] * 3
        )

        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)

        path = Path(__file__).parent.parent / "assets" / "models"
        p.setAdditionalSearchPath(str(path), physicsClientId=self._client)

        self._table_init_pos = [0.8, 0, 0]
        self._table_init_ori = [0, 0, 0, 1]
        self._obj_init_pos = [0.5, 0, 0.75]
        self._obj_init_ori = [0, 0, 0, 1]

        self._table = p.loadURDF(
            "table/table.urdf",
            self._table_init_pos,
            self._table_init_ori,
            physicsClientId=self._client,
        )
        self._obj = p.loadURDF(
            "cube/cube.urdf",
            self._obj_init_pos,
            self._obj_init_ori,
            physicsClientId=self._client,
        )

        # Let things fall down
        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)

        self.joints_initial_pose = self._robot.getAnglesPosition(
            self._robot.joint_dict.keys()
        )

        self._obj_start_pos = p.getBasePositionAndOrientation(
            self._obj, physicsClientId=self._client
        )[0]

    def _reset_scene(self):
        p.resetBasePositionAndOrientation(
            self._robot.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )

        self._reset_joint_state()

        p.resetBasePositionAndOrientation(
            self._table,
            posObj=self._table_init_pos,
            ornObj=self._table_init_ori,
            physicsClientId=self._client,
        )

        obj_pos = self._sample_goal()

        p.resetBasePositionAndOrientation(
            self._obj,
            posObj=obj_pos,
            ornObj=self._obj_init_ori,
            physicsClientId=self._client,
        )

        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        return self._get_observation()

    def _reset_joint_state(self):
        for joint, position in zip(
            self._robot.joint_dict.keys(), self.joints_initial_pose
        ):
            p.setJointMotorControl2(
                self._robot.robot_model,
                self._robot.joint_dict[joint].getIndex(),
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                physicsClientId=self._client,
            )
            p.resetJointState(
                self._robot.robot_model,
                self._robot.joint_dict[joint].getIndex(),
                position,
                physicsClientId=self._client,
            )

    def _get_observation(self):
        goal_pos = self._goal
        joint_p = self._robot.getAnglesPosition(CONTROLLABLE_JOINTS)
        joint_v = self._robot.getAnglesVelocity(CONTROLLABLE_JOINTS)
        hand_idx = self._robot.link_dict["l_hand"].getIndex()
        hand_pos = p.getLinkState(
            self._robot.getRobotModel(), hand_idx, physicsClientId=self._client
        )[0]
        obj_rel_pos = np.array(goal_pos) - np.array(hand_pos)

        return {
            "observation": np.concatenate(
                [goal_pos, joint_p, joint_v, obj_rel_pos]
            ).astype(np.float32),
            "achieved_goal": np.array(hand_pos, dtype=np.float32),
            "desired_goal": goal_pos,
        }

    def _sample_goal(self):
        return np.append(
            (
                np.random.sample(2) * [0.2, 0.4] + self._obj_init_pos[:2] - [0.25, 0.25]
            ).astype(np.float32),
            self._obj_init_pos[2],
        )

    def _is_table_displaced(self):
        pose = p.getBasePositionAndOrientation(
            self._table, physicsClientId=self._client
        )
        current_pose = np.array([e for t in pose for e in t], dtype=np.float32)
        desired_pose = np.concatenate([self._table_init_pos, self._table_init_ori])

        return not np.allclose(desired_pose, current_pose, atol=0.01)
