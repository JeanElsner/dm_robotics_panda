"""Panda Hand gripper."""
import enum
from typing import Dict, Sequence

import numpy as np
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from dm_robotics.moma import sensor
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.models import types
from dm_robotics.moma.models import utils as models_utils
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand

from . import gripper_constants as consts
from . import parameters as params


class PandaHand(robot_hand.RobotHand):
  """MoMa robot hand model of the Franka Hand."""

  _mjcf_root: mjcf.RootElement
  _state: consts.STATES

  def _build(self, name: str = 'panda_hand'):
    self._mjcf_root = mjcf.from_path(consts.XML_PATH)
    self._mjcf_root.model = name

    self._actuators = [
        self._mjcf_root.find('actuator', j) for j in consts.ACTUATOR_NAMES
    ]
    self._joints = [
        self._mjcf_root.find('joint', j) for j in consts.JOINT_NAMES
    ]
    self._tool_center_point = self._mjcf_root.find('site',
                                                   consts.GRIPPER_SITE_NAME)

    self._state = consts.STATES.READY

  @property
  def joints(self) -> Sequence[types.MjcfElement]:
    """List of joint elements belonging to the hand."""
    return self._joints

  @property
  def actuators(self) -> Sequence[types.MjcfElement]:
    """List of actuator elements belonging to the hand."""
    return self._actuators

  @property
  def mjcf_model(self) -> types.MjcfElement:
    """Returns the `mjcf.RootElement` object corresponding to the robot hand."""
    return self._mjcf_root

  @property
  def name(self) -> str:
    """Name of the robot hand."""
    return self.mjcf_model.model

  @property
  def tool_center_point(self) -> types.MjcfElement:
    """Tool center point site of the hand."""
    return self._tool_center_point

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState):
    """Function called at the beginning of every episode."""
    del random_state  # Unused.
    self._state = consts.STATES.READY

  def set_width(self, physics: mjcf.Physics, width: float):
    """Set desired aperture of the gripper."""
    self.set_joint_positions(physics, [width * 0.5] * 2)

  def set_joint_positions(self, physics: mjcf.Physics,
                          joint_positions: np.ndarray) -> None:
    """Sets the joints of the gripper to a given configuration.

    Changes the joint state as well as the current control signal to
    the desired joint state.

    Args:
      physics: A `mujoco.Physics` instance.
      joint_positions: The desired joint state for the robot gripper.
    """
    physics_joints = models_utils.binding(physics, self._joints)
    physics_actuators = models_utils.binding(physics, self._actuators)

    physics_joints.qpos[:] = joint_positions
    physics_actuators.ctrl[:] = joint_positions[0] * 2 / 0.08
    self._state = consts.STATES.READY


@enum.unique
class _PandaHandObservations(enum.Enum):
  WIDTH = '{}_width'

  STATE = '{}_state'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


class PandaHandSensor(sensor.Sensor):
  """Sensor for the Panda gripper.

  Provides two observations, namely, the current aperture or width
  between the fingers as well as gripper's state.
  """

  def __init__(self, gripper: PandaHand, name: str) -> None:
    self._gripper = gripper
    self._name = name
    self._observables = {
        self.get_obs_key(_PandaHandObservations.WIDTH):
            observable.Generic(self._width),
        self.get_obs_key(_PandaHandObservations.STATE):
            observable.Generic(self._state)
    }
    for obs in self._observables.values():
      obs.enabled = True
    self._threshold_samples = 0

  @property
  def name(self) -> str:
    return self._name

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  def get_obs_key(self, obs: enum.Enum) -> str:
    return obs.get_obs_key(self.name)

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  def _width(self, physics: mjcf.Physics) -> np.ndarray:
    return np.sum(physics.bind(self._gripper.joints).qpos)

  def _state(self, physics: mjcf.Physics) -> np.ndarray:
    speed = np.abs(np.sum(physics.bind(self._gripper.joints).qvel))
    if speed < consts.SPEED_TOLERANCE:
      if self._threshold_samples < consts.CONSECUTIVE_SAMPLES:
        self._threshold_samples += 1
      else:
        return np.array(consts.STATES.READY.value)
    else:
      self._threshold_samples = 0
    return np.array(consts.STATES.WAITING.value)


class PandaHandEffector(default_gripper_effector.DefaultGripperEffector):
  """Panda gripper effector.

  Action space is binary. The gripper's finger will move until blocked
  and then continuously apply a force either outwards or inwards, corresponding
  to actions 1 and 0 respectively. Actions between 0 and 1 get rounded."""

  def __init__(self, robot_params: params.RobotParams,
               gripper: robot_hand.RobotHand,
               panda_hand_sensor: PandaHandSensor):
    super().__init__(gripper, robot_params.name)
    self._robot_params = robot_params
    self._panda_hand_sensor = panda_hand_sensor
    self._state_getter = self._panda_hand_sensor.observables[
        self._panda_hand_sensor.get_obs_key(_PandaHandObservations.STATE)]
    self._spec = None

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    state = self._state_getter(physics)
    if state == consts.STATES.READY.value:
      command[0] = 0 if command[0] < 0.5 else 1
      super().set_control(physics, command)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    if self._spec is None:
      # self._spec = specs.DiscreteArray(2, name=f'{self.prefix}_grasp')
      self._spec = specs.BoundedArray((1,), np.float32, (0,), (1,),
                                      f'{self.prefix}_grasp')
    return self._spec


class DummyHand(robot_hand.RobotHand):
  """A fully MoMa compatible but empty RobotHand model."""

  def _build(self, name: str = 'dummy_hand'):
    self._mjcf_root = mjcf.RootElement()
    self._tool_center_point = self.mjcf_model.worldbody.add('site')
    self._mjcf_root.model = name

  @property
  def mjcf_model(self) -> types.MjcfElement:
    return self._mjcf_root

  @property
  def actuators(self) -> Sequence[types.MjcfElement]:
    return []

  @property
  def joints(self) -> Sequence[types.MjcfElement]:
    return []

  @property
  def name(self) -> str:
    return self._mjcf_root.model

  @property
  def tool_center_point(self) -> types.MjcfElement:
    return self._tool_center_point
