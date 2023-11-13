"""RL environment builder for the Panda MoMa model.
  This module allows you to build a full RL environment,
  including fully configured arena, initializers, robot and task.
  As such it can be considered the main module for users to
  interact with. The default configuration is a ready to run
  `dm_control.rl.control.Environment` but all aspects like e.g.
  reward, observation and action spec can be fully customized."""
import dataclasses
from typing import Callable, Optional, Sequence, Union

from dm_control import composer, mjcf
from dm_control.rl import control
from dm_robotics.agentflow.preprocessors import (observation_transforms,
                                                 timestep_preprocessor)
from dm_robotics.geometry import joint_angles_distribution, pose_distribution
from dm_robotics.moma import base_task, entity_initializer, prop
from dm_robotics.moma import robot as robot_module
from dm_robotics.moma import (scene_initializer, subtask_env,
                              subtask_env_builder)
from dm_robotics.moma.models.arenas import empty

from dm_robotics import agentflow as af

from . import arm, hardware
from . import parameters as params
from . import utils


class UserArena(empty.Arena):

  def _build(self, root_element: mjcf.RootElement):
    self._mjcf_root = root_element
    self._ground = self._mjcf_root.find('geom', 'ground')


class PandaEnvironmentBuilder:
  """Subtask environment builder class for the Panda MoMa model.

  Execution order:

    * build_arena
    * build_robot
    * build_entity_initializer
    * build_scene_initializer"""

  def __init__(
      self,
      robot_params: Union[params.RobotParams,
                          Sequence[params.RobotParams]] = params.RobotParams(),
      env_params: params.EnvirontmentParameters = params.EnvirontmentParameters(
      ),
      builder_extensions: params.BuilderExtensions = params.BuilderExtensions(),
      timestep_preprocessors: Optional[Sequence[
          timestep_preprocessor.TimestepPreprocessor]] = None
  ) -> None:
    self._robot_params = robot_params if isinstance(
        robot_params, Sequence) else [robot_params]
    self._env_params = env_params
    self._builder_extensions = builder_extensions
    self._timestep_preprocessors = timestep_preprocessors

  def build_task_environment(self) -> subtask_env.SubTaskEnvironment:
    """Builds an rl environment for the Panda robot."""
    task = self._build_task()

    env_builder = subtask_env_builder.SubtaskEnvBuilder()
    env_builder.set_task(task)

    base_env = env_builder.build_base_env()
    physics = base_env.physics

    parent_action_spec = task.effectors_action_spec(physics)
    full_action_space = af.IdentityActionSpace(parent_action_spec)

    if self._timestep_preprocessors is not None:
      for preproc in self._timestep_preprocessors:
        env_builder.add_preprocessor(preproc)

    env_builder.set_action_space(full_action_space)
    env = env_builder.build()

    return env

  def _bulid_arena(self) -> composer.Arena:
    arena = empty.Arena()
    if self._env_params.arena is not None:
      arena = UserArena(self._env_params.arena)
      # arena.mjcf_model.include_copy(self._env_params.arena,
      #                               override_attributes=True)
    arena.mjcf_model.compiler.angle = 'degree'
    arena.mjcf_model.option.timestep = 0.002
    if self._builder_extensions.build_arena is not None:
      self._builder_extensions.build_arena(arena)
    return arena

  def _build_scene_initializer(
      self, robots: Sequence[robot_module.Robot],
      arena: composer.Arena) -> base_task.SceneInitializer:
    scene_initializers = []
    if self._builder_extensions.build_scene_initializer is not None:
      scene_initializers.append(
          self._builder_extensions.build_scene_initializer(robots, arena))
    for robot, robot_params in zip(robots, self._robot_params):
      if robot_params.pose is not None:
        arm_pose = pose_distribution.ConstantPoseDistribution(robot_params.pose)
        scene_initializers.append(
            scene_initializer.EntityPoseInitializer(robot.arm_frame,
                                                    arm_pose.sample_pose))
    return scene_initializer.CompositeSceneInitializer(scene_initializers)

  def _build_entity_initializer(
      self, robots: Sequence[robot_module.Robot],
      arena: composer.Arena) -> entity_initializer.base_initializer.Initializer:
    entity_initializers = []
    if self._builder_extensions.build_entity_initializer is not None:
      entity_initializers.append(
          self._builder_extensions.build_entity_initializer(robots, arena))
    for robot, robot_params in zip(robots, self._robot_params):
      arm_joint_values = joint_angles_distribution.ConstantPanTiltDistribution(
          robot_params.joint_values)
      arm_init = entity_initializer.JointsInitializer(
          robot.position_arm_joints, arm_joint_values.sample_angles)
      entity_initializers.append(arm_init)
      if robot_params.has_hand:
        gripper_joint_values = joint_angles_distribution.ConstantPanTiltDistribution(
            [.04, .04])
        initialize_gripper = entity_initializer.JointsInitializer(
            robot.gripper.set_joint_angles, gripper_joint_values.sample_angles)
        entity_initializers.append(initialize_gripper)
    return entity_initializer.TaskEntitiesInitializer(entity_initializers)

  def _build_robot(self) -> Sequence[robot_module.Robot]:
    robots = []
    for robot_params in self._robot_params:
      if robot_params.robot_ip is not None:
        robot = hardware.build_robot(robot_params)
      else:
        robot = arm.build_robot(robot_params)
      robots.append(robot)
    if self._builder_extensions.build_robots is not None:
      self._builder_extensions.build_robots(robots)
    return robots

  def _build_props(self) -> Sequence[prop.Prop]:
    return []

  def _build_task(self) -> base_task.BaseTask:
    arena = self._bulid_arena()
    robots = self._build_robot()

    for robot, robot_params in zip(robots, self._robot_params):
      arena.attach(robot.arm, robot_params.attach_site)

    init_episode = self._build_entity_initializer(robots, arena)
    init_scene = self._build_scene_initializer(robots, arena)

    extra_effectors = []

    if self._builder_extensions.build_extra_effectors is not None:
      extra_effectors = self._builder_extensions.build_extra_effectors(
          robots, arena)

    # block = prop.Block()
    # # frame = arena.attach(block)
    # frame = arena.add_free_entity(block)
    # block.set_freejoint(frame.freejoint)

    # gripper_pose_dist = pose_distribution.UniformPoseDistribution(
    #     # Provide 6D min and max bounds for the 3D position and 3D euler angles.
    #     min_pose_bounds=np.array(
    #         [0.5, -0.1, 0.1, 0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi]),
    #     max_pose_bounds=np.array(
    #         [0.7, 0.1, 0.2, 1.25 * np.pi, 0.25 * np.pi, 0.5 * np.pi]))

    # initialize_arm = entity_initializer.PoseInitializer(
    #     moma_robot.position_gripper, gripper_pose_dist.sample_pose)

    # block_pose_dist = pose_distribution.UniformPoseDistribution(
    #     min_pose_bounds=[0.1, -0.5, 0.02, 0, 0, 0],
    #     max_pose_bounds=[0.5, 0.5, 0.02, 0, 0, 0])
    # initialize_block = entity_initializer.PoseInitializer(
    #     block.set_pose, block_pose_dist.sample_pose)
    # init_episode = entity_initializer.TaskEntitiesInitializer([init_episode, initialize_block])

    task = base_task.BaseTask(
        task_name='panda',
        arena=arena,
        robots=robots,
        props=[],
        extra_sensors=[utils.TimeSensor()],
        extra_effectors=extra_effectors,
        control_timestep=self._env_params.control_timestep,
        scene_initializer=init_scene,
        episode_initializer=init_episode)
    return task
