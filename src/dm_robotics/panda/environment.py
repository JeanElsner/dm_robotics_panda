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
from dm_robotics.moma import (scene_initializer, sensor, subtask_env,
                              subtask_env_builder)
from dm_robotics.moma.models.arenas import empty

from dm_robotics import agentflow as af

from . import arm, hardware
from . import parameters as params
from . import utils


class PandaEnvironment:

  def __init__(self,
               robot_params: Union[
                   params.RobotParams,
                   Sequence[params.RobotParams]] = params.RobotParams(),
               arena: composer.Arena = empty.Arena(),
               control_timestep: float = 0.1,
               physics_timestep: float = 0.002) -> None:
    self._arena = arena
    self._robot_params = robot_params if isinstance(
        robot_params, Sequence) else [robot_params]
    self._control_timestep = control_timestep

    self._arena.mjcf_model.compiler.angle = 'degree'
    self._arena.mjcf_model.option.timestep = physics_timestep

    self._robots = []
    for robot_params in self._robot_params:
      if robot_params.robot_ip is not None:
        robot = hardware.build_robot(robot_params)
      else:
        robot = arm.build_robot(robot_params)
      self._robots.append(robot)
    for robot, robot_params in zip(self._robots, self._robot_params):
      self._arena.attach(robot.arm, robot_params.attach_site)

    self._scene_initializers = []
    self._entity_initializers = []
    self._extra_sensors = []
    self._extra_effectors = []
    self._props = []
    self._timestep_preprocessors = []

  @property
  def robots(self) -> Sequence[robot_module.Robot]:
    return self._robots

  def build_task_environment(self) -> subtask_env.SubTaskEnvironment:
    """Builds an rl environment for the Panda robot."""
    self._extra_sensors.append(utils.TimeSensor())
    task = base_task.BaseTask(
        task_name='panda',
        arena=self._arena,
        robots=self._robots,
        props=self._props,
        extra_sensors=self._extra_sensors,
        extra_effectors=self._extra_effectors,
        control_timestep=self._control_timestep,
        scene_initializer=self._build_scene_initializer(),
        episode_initializer=self._build_entity_initializer())

    env_builder = subtask_env_builder.SubtaskEnvBuilder()
    env_builder.set_task(task)

    base_env = env_builder.build_base_env()
    physics = base_env.physics

    parent_action_spec = task.effectors_action_spec(physics)
    full_action_space = af.IdentityActionSpace(parent_action_spec)

    for preproc in self._timestep_preprocessors:
      env_builder.add_preprocessor(preproc)

    env_builder.set_action_space(full_action_space)
    env = env_builder.build()

    return env

  def add_props(self, props: Sequence[prop.Prop]):
    self._props.extend(props)
    for p in self._props:
      self._arena.add_free_entity(p)

  def add_entity_initializers(
      self,
      initializers: Sequence[entity_initializer.base_initializer.Initializer]):
    self._entity_initializers.extend(initializers)

  def add_timestep_preprocessors(
      self,
      preprocessors: Sequence[timestep_preprocessor.TimestepPreprocessor]):
    self._timestep_preprocessors.extend(preprocessors)

  def add_extra_sensors(self, extra_sensors: Sequence[sensor.Sensor]):
    self._extra_sensors.extend(extra_sensors)

  def _build_scene_initializer(self) -> base_task.SceneInitializer:
    for robot, robot_params in zip(self._robots, self._robot_params):
      if robot_params.pose is not None:
        arm_pose = pose_distribution.ConstantPoseDistribution(robot_params.pose)
        self._scene_initializers.insert(
            0,
            scene_initializer.EntityPoseInitializer(robot.arm_frame,
                                                    arm_pose.sample_pose))
    return scene_initializer.CompositeSceneInitializer(self._scene_initializers)

  def _build_entity_initializer(
      self) -> entity_initializer.base_initializer.Initializer:
    for robot, robot_params in zip(self._robots, self._robot_params):
      arm_joint_values = joint_angles_distribution.ConstantPanTiltDistribution(
          robot_params.joint_values)
      arm_init = entity_initializer.JointsInitializer(
          robot.position_arm_joints, arm_joint_values.sample_angles)
      self._entity_initializers.insert(0, arm_init)
      if robot_params.has_hand:
        gripper_joint_values = joint_angles_distribution.ConstantPanTiltDistribution(
            [.04, .04])
        initialize_gripper = entity_initializer.JointsInitializer(
            robot.gripper.set_joint_angles, gripper_joint_values.sample_angles)
        self._entity_initializers.insert(0, initialize_gripper)
    return entity_initializer.TaskEntitiesInitializer(self._entity_initializers)
