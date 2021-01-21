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


class PepperPushCamEnv(gym.GoalEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        gui=False,
        sim_steps_per_action=10,
        max_motion_speed=0.3,
        use_depth_camera=False,
        use_top_camera=False,
    ):
        self._sim_steps = sim_steps_per_action
        self._max_speeds = [max_motion_speed] * len(CONTROLLABLE_JOINTS)
        self._gui = gui
        self._use_depth_camera = use_depth_camera
        self._use_top_camera = use_top_camera

        self._setup_scene()

        self._goal = self._sample_goal()
        obs = self._get_observation()

        self.action_space = spaces.Box(
            -2.0857, 2.0857, shape=(len(CONTROLLABLE_JOINTS),), dtype="float32"
        )

        obs_spaces = dict(
            camera_bottom=spaces.Box(
                0,
                255,
                shape=obs["camera_bottom"].shape,
                dtype=obs["camera_bottom"].dtype,
            ),
            joints_state=spaces.Box(
                -np.inf,
                np.inf,
                shape=obs["joints_state"].shape,
                dtype=obs["joints_state"].dtype,
            ),
            object_position=spaces.Box(
                -np.inf,
                np.inf,
                shape=obs["object_position"].shape,
                shape=obs["object_position"].dtype,
            )
        )

        if self._use_top_camera:
            obs_spaces["camera_top"] = (
                spaces.Box(
                    0,
                    255,
                    shape=obs["camera_top"].shape,
                    dtype=obs["camera_top"].dtype,
                ),
            )

        if self._use_depth_camera:
            obs_spaces["camera_depth"] = spaces.Box(
                0,
                65_535,
                shape=obs["camera_depth"].shape,
                dtype=obs["camera_depth"].dtype,
            )

        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Dict(obs_spaces),
            )
        )

    def reset(self):
        self._reset_scene()
        self._goal = self._sample_goal()

        if self._gui:
            self._place_ghosts()

        return self._get_observation()

    def step(self, action):
        action = list(action)
        assert len(action) == len(self.action_space.high.tolist())

        self._robot.setAngles(CONTROLLABLE_JOINTS, action, self._max_speeds)

        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_observation()

        is_success = self._is_success(obs["achieved_goal"], self._goal)
        is_safety_violated = self._is_table_touched() or self._is_table_displaced()

        info = {
            "is_success": is_success,
        }
        reward = self.compute_reward(obs["achieved_goal"], self._goal, info)
        done = is_success or is_safety_violated

        return (obs, reward, done, info)

    def render(self, mode="human"):
        pass

    def close(self):
        if self._use_top_camera:
            self._robot.unsubscribeCamera(self._cam_top)
        self._robot.unsubscribeCamera(self._cam_bottom)
        if self._use_depth_camera:
            self._robot.unsubscribeCamera(self._cam_depth)
        self._simulation_manager.stopSimulation(self._client)

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return (
            np.linalg.norm(desired_goal - achieved_goal, axis=-1) < DISTANCE_THRESHOLD
        ).astype(np.float32)

    def _is_success(self, achieved_goal, desired_goal):
        return (
            np.linalg.norm(desired_goal - achieved_goal, axis=-1) < DISTANCE_THRESHOLD
        )

    def _setup_scene(self):
        self._simulation_manager = SimulationManager()
        self._client = self._simulation_manager.launchSimulation(
            gui=self._gui, auto_step=False
        )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self._robot = self._simulation_manager.spawnPepper(
            self._client, spawn_ground_plane=False
        )

        self._robot.goToPosture("Stand", 1.0)

        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)

        self._robot.setAngles(
            ["KneePitch", "HipPitch", "LShoulderPitch"], [0.33, -0.9, -0.6], [0.5] * 3
        )

        for _ in range(500):
            p.stepSimulation(physicsClientId=self._client)

        self._table_init_pos = [0.35, 0, 0]
        self._table_init_ori = [0, 0, 0, 1]
        self._obj_init_pos = [0.35, 0, 0.65]
        self._obj_init_ori = [0, 0, 0, 1]

        path = Path(__file__).parent.parent / "assets" / "models"
        p.setAdditionalSearchPath(str(path), physicsClientId=self._client)

        self._floor = p.loadURDF(
            "floor/floor.urdf", physicsClientId=self._client, useFixedBase=True
        )

        self._table = p.loadURDF(
            "adjustable_table/adjustable_table.urdf",
            self._table_init_pos,
            self._table_init_ori,
            physicsClientId=self._client,
        )
        self._obj = p.loadURDF(
            "brick/brick.urdf",
            self._obj_init_pos,
            self._obj_init_ori,
            physicsClientId=self._client,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
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

        if self._gui:
            # load ghosts
            self._ghost = p.loadURDF(
                "brick/brick_ghost.urdf",
                self._obj_start_pos,
                self._obj_init_ori,
                physicsClientId=self._client,
                useFixedBase=True,
            )

        # Setup camera
        if self._use_top_camera:
            self._cam_top = self._robot.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)
        self._cam_bottom = self._robot.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)
        if self._use_depth_camera:
            self._cam_depth = self._robot.subscribeCamera(PepperVirtual.ID_CAMERA_DEPTH)

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
        obj_pos = p.getBasePositionAndOrientation(
            self._obj, physicsClientId=self._client
        )[0]

        img_bottom = self._robot.getCameraFrame(self._cam_bottom)

        joint_p = self._robot.getAnglesPosition(CONTROLLABLE_JOINTS)
        # joint_v = self._robot.getAnglesVelocity(CONTROLLABLE_JOINTS)

        result = {
            "object_position": obj_pos,
            "camera_bottom": img_bottom,
            # "joints_state": np.concatenate([joint_p, joint_v]).astype(np.float32),
            "joints_state": np.array(joint_p, dtype=np.float32)
        }

        if self._use_top_camera:
            img_top = self._robot.getCameraFrame(self._cam_top)
            result["camera_top"] = img_top

        if self._use_depth_camera:
            img_depth = self._robot.getCameraFrame(self._cam_depth)
            result["camera_depth"] = img_depth

        return {
            "observation": result,
            "achieved_goal": np.array(obj_pos, dtype=np.float32),
            "desired_goal": self._goal,
        }

    def _sample_goal(self):
        return np.append(
            (
                np.random.sample(2) * [0.2, 0.4] + self._obj_start_pos[:2] - [0.1, 0.2]
            ).astype(np.float32),
            self._obj_start_pos[2],
        )

    def _is_table_displaced(self):
        pose = p.getBasePositionAndOrientation(
            self._table, physicsClientId=self._client
        )
        current_pose = np.array([e for t in pose for e in t], dtype=np.float32)
        desired_pose = np.concatenate([self._table_init_pos, self._table_init_ori])

        return not np.allclose(desired_pose, current_pose, atol=0.01)

    def _is_table_touched(self):
        cont = p.getContactPoints(self._robot.getRobotModel(), self._table)

        return len(cont) > 0

    def _place_ghosts(self):
        p.resetBasePositionAndOrientation(
            self._ghost,
            posObj=self._goal,
            ornObj=self._obj_init_ori,
            physicsClientId=self._client,
        )