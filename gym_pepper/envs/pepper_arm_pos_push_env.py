# Some bits are based on:
# https://github.com/softbankrobotics-research/qi_gym/blob/master/envs/throwing_env.py

from pathlib import Path

import gym
import numpy as np
import pybullet as p
from gym import error, spaces, utils
from gym.utils import seeding
from qibullet import PepperVirtual, SimulationManager

TIME_STEP = 0.1
DISTANCE_THRESHOLD = 0.04
WORKSPACE_RADIUS = 2.0
CONTROLLABLE_JOINTS = [
    "LShoulderPitch",
    "LShoulderRoll",
    "LElbowYaw",
    "LElbowRoll",
    "LWristYaw",
    "LHand",
]
CONSTANT_JOINTS = [
    "KneePitch",
    "HipPitch",
    "HipRoll",
    "HeadYaw",
    "HeadPitch",
]


class PepperArmPosPushEnv(gym.GoalEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, gui=False, sim_steps_per_action=10, max_motion_speed=0.5):
        self._sim_steps = sim_steps_per_action
        self._max_speeds = [max_motion_speed] * 6
        self._gui = gui

        self._setup_scene()

        self._goal_xy = self._sample_goal()
        obs = self._get_observation()

        self.action_space = spaces.Box(-2.0857, 2.0857, shape=(6,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape, dtype='float32'),
        ))

    def reset(self):
        self._reset_scene()
        self._goal_xy = self._sample_goal()

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

        is_success = self._is_success(obs['achieved_goal'], self._goal_xy)
        info = {
            'is_success': is_success,
        }
        reward = self.compute_reward(
            obs['achieved_goal'], self._goal_xy, info)
        done = is_success or self._is_table_displaced()

        return (obs, reward, done, info)

    def render(self, mode='human'):
        pass

    def close(self):
        self._simulation_manager.stopSimulation(self._client)

    def seed(self, seed=None):
        np.random.seed(seed or 0)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return (np.linalg.norm(desired_goal - achieved_goal, axis=-1) < DISTANCE_THRESHOLD).astype(np.float32)

    def _setup_scene(self):
        self._simulation_manager = SimulationManager()
        self._client = self._simulation_manager.launchSimulation(gui=self._gui, auto_step=False)
        
        self._robot = self._simulation_manager.spawnPepper(
            self._client, spawn_ground_plane=True)

        self._robot.goToPosture("Stand", 1.0)

        for _ in range(1000):
            p.stepSimulation(physicsClientId=self._client)

        self._robot.setAngles(CONSTANT_JOINTS, [0.01, -1.0, 0, 0, 0,], [self._max_speed] * 5)
        self._robot.setAngles(CONTROLLABLE_JOINTS, [0, 0, 0, -1.5, 0, 1.], [self._max_speed] * 6)

        self.joints_initial_pose = self._robot.getAnglesPosition(
            self._robot.joint_dict.keys())

        path = Path(__file__).parent.parent / "assets" / "models"
        p.setAdditionalSearchPath(str(path), physicsClientId=self._client)

        self._lack_initial_position = [0.5, 0, 0]
        self._lack_initial_orientation = [0, 0, 0, 1]
        self._cube_initial_position = [0.5, 0, 0.6]
        self._cube_initial_orientation = [0, 0, 0, 1]

        self._lack = p.loadURDF(
            "Lack/Lack.urdf", self._lack_initial_position, self._lack_initial_orientation, 
            physicsClientId=self._client)
        self._cube = p.loadURDF(
            "cube/cube.urdf", self._cube_initial_position, self._cube_initial_orientation, 
            physicsClientId=self._client)

        if self._gui:
            # load ghosts
            self._cube_ghost = p.loadURDF(
                "cube/cube_ghost.urdf", self._cube_initial_position, self._cube_initial_orientation, 
                physicsClientId=self._client, useFixedBase=True)

        for _ in range(1000):
            p.stepSimulation(physicsClientId=self._client)

    def _reset_scene(self):
        p.resetBasePositionAndOrientation(
            self._robot.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client)

        self._reset_joint_state()

        p.resetBasePositionAndOrientation(
            self._lack,
            posObj=self._lack_initial_position,
            ornObj=self._lack_initial_orientation,
            physicsClientId=self._client)
        p.resetBasePositionAndOrientation(
            self._cube,
            posObj=self._cube_initial_position,
            ornObj=self._cube_initial_orientation,
            physicsClientId=self._client)

        for _ in range(self._sim_steps):
            p.stepSimulation(physicsClientId=self._client)

        return self._get_observation()

    def _reset_joint_state(self):
        for joint, position in\
                zip(self._robot.joint_dict.keys(), self.joints_initial_pose):
            p.setJointMotorControl2(
                self._robot.robot_model,
                self._robot.joint_dict[joint].getIndex(),
                p.VELOCITY_CONTROL,
                targetVelocity=0.0, 
                physicsClientId=self._client)
            p.resetJointState(
                self._robot.robot_model,
                self._robot.joint_dict[joint].getIndex(),
                position,
                physicsClientId=self._client)

    def _get_observation(self):
        obj_pos = p.getBasePositionAndOrientation(self._cube, physicsClientId=self._client)[0]
        obj_vel = p.getBaseVelocity(self._cube, physicsClientId=self._client)[0]
        joint_p = self._robot.getAnglesPosition(CONTROLLABLE_JOINTS)
        joint_v = self._robot.getAnglesVelocity(CONTROLLABLE_JOINTS)
        hand_idx = self._robot.link_dict["l_hand"].getIndex()
        hand_pos = p.getLinkState(self._robot.getRobotModel(), hand_idx, physicsClientId=self._client)[0]
        obj_rel_pos = np.array(obj_pos) - np.array(hand_pos)

        return {
            'observation': np.concatenate([obj_pos, obj_vel, joint_p, joint_v, obj_rel_pos]).astype(np.float32),
            'achieved_goal': np.array(obj_pos[:2], dtype=np.float32),
            'desired_goal': self._goal_xy
        }

    def _is_success(self, achieved_goal, desired_goal):
        return np.linalg.norm(desired_goal - achieved_goal, axis=-1) < DISTANCE_THRESHOLD

    def _sample_goal(self):
        return (np.random.sample(2) * [0.5, 0.5] + self._lack_initial_position[:2] - [0.25, 0.25]).astype(np.float32)

    def _is_table_displaced(self):
        pose = p.getBasePositionAndOrientation(self._lack, physicsClientId=self._client)
        current_pose = np.array([e for t in pose for e in t], dtype=np.float32)
        desired_pose = np.concatenate(
            [self._lack_initial_position, self._lack_initial_orientation])

        return not np.allclose(desired_pose, current_pose, atol=0.01)

    def _place_ghosts(self):
        cube_pose = p.getBasePositionAndOrientation(self._cube, physicsClientId=self._client)
        p.resetBasePositionAndOrientation(
            self._cube_ghost,
            posObj=list(self._goal_xy) + [cube_pose[0][2]],
            ornObj=self._cube_initial_orientation,
            physicsClientId=self._client)
