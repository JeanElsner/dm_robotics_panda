"""RL environment builder for the Panda MoMa model."""
import collections
import typing
from typing import Sequence, Union

from dm_control import composer
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.geometry import joint_angles_distribution, pose_distribution
from dm_robotics.moma import base_task, effector, entity_initializer, prop
from dm_robotics.moma import robot as robot_module
from dm_robotics.moma import (scene_initializer, sensor, subtask_env,
                              subtask_env_builder)
from dm_robotics.moma.models.arenas import empty

from dm_robotics import agentflow as af

from . import arm, hardware
from . import parameters as params
from . import utils


class PandaEnvironment:
  """Adds Panda robots to an arena and creates a subtask environment."""

  def __init__(self,
               robot_params: Union[params.RobotParams,
                                   Sequence[params.RobotParams]],
               arena: composer.Arena = empty.Arena(),
               control_timestep: float = 0.1,
               physics_timestep: float = 0.002) -> None:
    self._arena = arena
    self._robot_params = robot_params if isinstance(
        robot_params, Sequence) else [robot_params]
    self._control_timestep = control_timestep
    self._arena.mjcf_model.option.timestep = physics_timestep

    self._robots = collections.OrderedDict()
    for robot_params in self._robot_params:
      if robot_params.robot_ip is not None:
        robot = hardware.build_robot(robot_params)
      else:
        robot = arm.build_robot(robot_params)
      self._robots[robot_params.name] = robot
    for robot, robot_params in zip(self._robots.values(), self._robot_params):
      self._arena.attach(robot.arm, robot_params.attach_site)

    self._scene_initializers = []
    self._entity_initializers = []
    self._extra_sensors = []
    self._extra_effectors = []
    self._props = []
    self._timestep_preprocessors = []

  @property
  def robots(self) -> typing.Dict[str, robot_module.Robot]:
    """A map of the configured robots indexed by name."""
    return self._robots

  def build_task_environment(self) -> subtask_env.SubTaskEnvironment:
    """Builds a subtask environment.
    
    The environment includes the configured robots as well as
    any added components such as props, initializers, preprocessors etc.
    """
    self._extra_sensors.append(utils.TimeSensor())
    task = base_task.BaseTask(
        task_name='panda',
        arena=self._arena,
        robots=self._robots.values(),
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
    """Add props as free entities to the arena."""
    self._props.extend(props)
    for p in self._props:
      frame = self._arena.add_free_entity(p)
      p.set_freejoint(frame.freejoint)

  def add_entity_initializers(
      self,
      initializers: Sequence[entity_initializer.base_initializer.Initializer]):
    """Add entity initializers."""
    self._entity_initializers.extend(initializers)

  def add_scene_initializers(
      self, initializers: Sequence[base_task.SceneInitializer]):
    """Add scene initializers."""
    self._scene_initializers.extend(initializers)

  def add_timestep_preprocessors(
      self,
      preprocessors: Sequence[timestep_preprocessor.TimestepPreprocessor]):
    """Add timestep preprocessors."""
    self._timestep_preprocessors.extend(preprocessors)

  def add_extra_sensors(self, extra_sensors: Sequence[sensor.Sensor]):
    """Add extra sensor."""
    self._extra_sensors.extend(extra_sensors)

  def add_extra_effectors(self, extra_effectors: Sequence[effector.Effector]):
    """Add extra effectors."""
    self._extra_effectors.extend(extra_effectors)

  def _build_scene_initializer(self) -> base_task.SceneInitializer:
    for robot, robot_params in zip(self._robots.values(), self._robot_params):
      if robot_params.pose is not None:
        arm_pose = pose_distribution.ConstantPoseDistribution(robot_params.pose)
        self._scene_initializers.insert(
            0,
            scene_initializer.EntityPoseInitializer(robot.arm_frame,
                                                    arm_pose.sample_pose))
    return scene_initializer.CompositeSceneInitializer(self._scene_initializers)

  def _build_entity_initializer(
      self) -> entity_initializer.base_initializer.Initializer:
    for robot, robot_params in zip(self._robots.values(), self._robot_params):
      arm_joint_values = joint_angles_distribution.ConstantPanTiltDistribution(
          robot_params.joint_positions)
      arm_init = entity_initializer.JointsInitializer(
          robot.position_arm_joints, arm_joint_values.sample_angles)
      self._entity_initializers.insert(0, arm_init)
      if robot_params.has_hand:
        gripper_joint_values = joint_angles_distribution.ConstantPanTiltDistribution(
            [.04, .04])
        initialize_gripper = entity_initializer.JointsInitializer(
            robot.gripper.set_joint_positions,
            gripper_joint_values.sample_angles)
        self._entity_initializers.insert(0, initialize_gripper)
    return entity_initializer.TaskEntitiesInitializer(self._entity_initializers)
