"""Contains dataclasses holding parameter configurations."""
import dataclasses
from typing import Callable, Optional, Sequence

from dm_control import composer, mjcf
from dm_robotics.moma import (base_task, effector, entity_initializer, robot,
                              sensor)
from dm_robotics.moma.models.arenas import empty

from . import arm_constants


@dataclasses.dataclass
class RobotParams:
  """Parameters used for building the Panda robot model.

  Args:
    name: Name of the robot, used as prefix in MoMa and namespace in `MJCF`.
    pose: 6 DOF pose (position, euler angles) of the Panda's base.
    joint_values: Initial joint values of the arm.
    attachment_frame: The frame the robot will be attached to.
      If `None` the `Arena`'s default attachment frame is used.
    control_frame: MoMa `Effector`s and `Sensor`s will use this frame
      as a reference where applicable or use world frame if `None`.
    actuation: Actuation mode.
    cartesian: Use `Effector` with cartesian commands.
    has_hand: Set to true to use the Panda Hand.
    robot_ip: Robot IP or hostname. If `None` hardware in the loop is not used.
    joint_stiffness: Joint stiffness of the robot used in actuation.
    joint_damping: Joint damping of the robot used in actuation.
  """
  name: str = 'panda'
  pose: Optional[Sequence[float]] = None
  joint_values: Sequence[float] = (0, -0.785, 0, -2.356, 0, 1.571, 0.785)
  attach_site: Optional[mjcf.Element] = None
  control_frame: Optional[mjcf.Element] = None
  actuation: arm_constants.Actuation = arm_constants.Actuation.CARTESIAN_VELOCITY
  has_hand: bool = True
  robot_ip: Optional[str] = None
  joint_stiffness: Sequence[float] = (600, 600, 600, 600, 250, 150, 50)
  joint_damping: Sequence[float] = (50, 50, 50, 20, 20, 20, 10)


@dataclasses.dataclass
class Extensions:
  """Defines callables to extend the subtask environment builder."""
  build_arena: Optional[Callable[[empty.Arena], None]] = None
  build_robots: Optional[Callable[[Sequence[robot.Robot]], None]] = None
  build_scene_initializer: Optional[
      Callable[[Sequence[robot.Robot], composer.Arena],
               base_task.SceneInitializer]] = None
  build_entity_initializer: Optional[
      Callable[[Sequence[robot.Robot], composer.Arena],
               entity_initializer.base_initializer.Initializer]] = None
  build_extra_effectors: Optional[
      Callable[[Sequence[robot.Robot], composer.Arena],
               Sequence[effector.Effector]]] = None
  build_extra_sensors: Optional[Callable[
      [Sequence[robot.Robot], composer.Arena], Sequence[sensor.Sensor]]] = None


@dataclasses.dataclass
class EnvirontmentParameters:
  """Task-level parameters.

  Args:
    mjcf_root: MJCF model of the scene. The Panda robots will be added
      by :py:class:`dm_robotics.panda.env_builder.PandaEnvironmentBuilder`-
    arena: Reinforcement learning environment.
    control_timestep: Timestep size of the controlling agent."""
  mjcf_root: Optional[mjcf.RootElement] = None
  arena: Optional[composer.Environment] = None
  control_timestep: float = 0.1
