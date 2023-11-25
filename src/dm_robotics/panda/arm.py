"""MoMa robot arm model of the Panda along with specialized effectors and sensors."""
import dataclasses
import enum
import logging
from typing import Dict, List, Sequence, Tuple

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_env import specs
from dm_robotics.geometry import geometry, mujoco_physics
from dm_robotics.moma import effector, robot, sensor
from dm_robotics.moma.effectors import (arm_effector,
                                        cartesian_6d_velocity_effector)
from dm_robotics.moma.models import types
from dm_robotics.moma.models import utils as models_utils
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.sensors import (robot_arm_sensor, robot_tcp_sensor,
                                      site_sensor, wrench_observations)
from dm_robotics.transformations import transformations as tr

from . import arm_constants as consts
from . import gripper as gripper_module
from . import parameters as params
from . import utils

log = logging.getLogger('arm')


@dataclasses.dataclass(frozen=True)
class _ActuatorParams:
  # Gain parameters for MuJoCo actuator.
  gainprm: Tuple[float]
  # Bias parameters for MuJoCo actuator.
  biasprm: Tuple[float, float, float]


_PANDA_ACTUATOR_PARAMS = {
    consts.Actuation.CARTESIAN_VELOCITY: [
        _ActuatorParams((600.0,), (0.0, -600.0, -50.0)),
        _ActuatorParams((600.0,), (0.0, -600.0, -50.0)),
        _ActuatorParams((600.0,), (0.0, -600.0, -50.0)),
        _ActuatorParams((600.0,), (0.0, -600.0, -20.0)),
        _ActuatorParams((250.0,), (0.0, -250.0, -20.0)),
        _ActuatorParams((150.0,), (0.0, -150.0, -20.0)),
        _ActuatorParams((50.0,), (0.0, -50.0, -10.0))
    ]
}
_PANDA_ACTUATOR_PARAMS[
    consts.Actuation.JOINT_VELOCITY] = _PANDA_ACTUATOR_PARAMS[
        consts.Actuation.CARTESIAN_VELOCITY]
_PANDA_ACTUATOR_PARAMS[consts.Actuation.HAPTIC] = _PANDA_ACTUATOR_PARAMS[
    consts.Actuation.CARTESIAN_VELOCITY]


class Panda(robot_arm.RobotArm):
  """A class representing a Panda robot arm.

  Args:
    name: The name of this robot. Used as a prefix in the MJCF name
      attributes.
    actuation: Instance of :py:class:`dm_robotics.panda.arm_constants.Actuation`
      specifying which actuation mode to use.
    use_rotated_gripper: If True, mounts the gripper in a rotated position to
      match the real placement of the gripper on the physical Panda.
  """

  # Define member variables that are created in the _build function. This is to
  # comply with pytype correctly.
  _joints: List[types.MjcfElement]
  _actuators: List[types.MjcfElement]
  _joint_torque_sensors: List[types.MjcfElement]
  _mjcf_root: mjcf.RootElement
  _actuation: consts.Actuation
  _base_site: types.MjcfElement
  _wrist_site: types.MjcfElement
  _attachment_site: types.MjcfElement

  def _build(self,
             name: str = 'panda',
             actuation: consts.Actuation = consts.Actuation.CARTESIAN_VELOCITY,
             use_rotated_gripper: bool = True,
             hardware=None) -> None:
    self.hardware = hardware
    self._mjcf_root = mjcf.from_path(consts.XML_PATH)
    self._mjcf_root.model = name
    self._actuation = actuation

    self._add_mjcf_elements(use_rotated_gripper)
    self._add_actuators()

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState):
    """Function called at the beginning of every episode."""
    del random_state  # Unused.

  @property
  def actuation(self) -> consts.Actuation:
    """Selected actuation mode."""
    return self._actuation

  @property
  def joints(self) -> List[types.MjcfElement]:
    """List of joint elements belonging to the arm."""
    if not self._joints:
      raise AttributeError('Robot joints is None.')
    return self._joints

  @property
  def actuators(self) -> List[types.MjcfElement]:
    """List of actuator elements belonging to the arm."""
    if not self._actuators:
      raise AttributeError('Robot actuators is None.')
    return self._actuators

  @property
  def joint_torque_sensors(self) -> List[types.MjcfElement]:
    """Get MuJoCo sensor of the joint torques."""
    return self._joint_torque_sensors

  @property
  def mjcf_model(self) -> mjcf.RootElement:
    """Returns the `mjcf.RootElement` object corresponding to this robot arm."""
    if not self._mjcf_root:
      raise AttributeError('Robot mjcf_root is None.')
    return self._mjcf_root

  @property
  def name(self) -> str:
    """Name of the robot arm."""
    return self.mjcf_model.model

  @property
  def base_site(self) -> types.MjcfElement:
    """Get the MuJoCo site of the base.

    Returns:
      MuJoCo site
    """
    return self._base_site

  @property
  def wrist_site(self) -> types.MjcfElement:
    """Get the MuJoCo site of the wrist.

    Returns:
      MuJoCo site
    """
    return self._wrist_site

  @property
  def attachment_site(self):
    """Override wrist site for attachment, but NOT the one for observations."""
    return self._attachment_site

  def set_joint_angles(self, physics: mjcf.Physics,
                       joint_angles: np.ndarray) -> None:
    """Sets the joints of the robot to a given configuration.

    This function allows to change the joint configuration of the Panda arm
    and sets the controller to prevent the impedance controller from moving back
    to the previous configuration.

    Args:
      physics: A `mujoco.Physics` instance.
      joint_angles: The desired joints configuration for the robot arm.
    """
    physics_joints = models_utils.binding(physics, self._joints)
    physics_actuators = models_utils.binding(physics, self._actuators)

    physics_joints.qpos[:] = joint_angles
    if self._actuation in [
        consts.Actuation.CARTESIAN_VELOCITY, consts.Actuation.JOINT_VELOCITY
    ]:
      physics_actuators.act[:] = physics_joints.qpos[:]
    elif self._actuation == consts.Actuation.HAPTIC:
      physics_actuators.ctrl[:] = physics_joints.qpos[:]

  def after_substep(self, physics: mjcf.Physics,
                    random_state: np.random.RandomState) -> None:
    del random_state  # Unused.

    # Clip the actuator.act with the actuator limits.
    if self._actuation in [
        consts.Actuation.CARTESIAN_VELOCITY, consts.Actuation.JOINT_VELOCITY
    ]:
      physics_actuators = models_utils.binding(physics, self._actuators)
      physics_actuators.act[:] = np.clip(physics_actuators.act[:],
                                         a_min=consts.JOINT_LIMITS['min'],
                                         a_max=consts.JOINT_LIMITS['max'])

  def _add_mjcf_elements(self, use_rotated_gripper: bool):
    self._joints = [
        self._mjcf_root.find('joint', j) for j in consts.JOINT_NAMES
    ]
    self._joint_torque_sensors = [
        self._mjcf_root.find('sensor', j)
        for j in consts.JOINT_TORQUE_SENSOR_NAMES
    ]
    self._base_site = self._mjcf_root.find('site', consts.BASE_SITE_NAME)
    self._wrist_site = self._mjcf_root.find('site', consts.WRIST_SITE_NAME)

    if use_rotated_gripper:
      # Change the attachment site so it is aligned with the real Panda. This
      # will allow having the gripper oriented in the same way in both sim and
      # real.
      hand_body = self._mjcf_root.find('body', 'panda_link8')
      hand_body.add('site',
                    type='sphere',
                    name='real_aligned_tcp',
                    pos=(0, 0, 0),
                    quat=consts.ROTATION_QUATERNION_MINUS_45DEG_AROUND_Z)
      self._attachment_site = self._mjcf_root.find('site', 'real_aligned_tcp')
    else:
      self._attachment_site = self._wrist_site

  def _add_actuators(self):
    if self._actuation not in consts.Actuation:
      raise ValueError((f'Actuation {self._actuation} is not a valid actuation.'
                        'Please specify one of '
                        f'{list(consts.Actuation.__members__.values())}'))

    if self._actuation in [
        consts.Actuation.CARTESIAN_VELOCITY, consts.Actuation.JOINT_VELOCITY
    ]:
      self._add_mjcf_actuators()
    elif self._actuation == consts.Actuation.HAPTIC:
      self._add_mjcf_actuators(dyntype='none')

  def _add_mjcf_actuators(self, dyntype: str = 'integrator') -> None:
    # Construct list of ctrlrange tuples from act limits and actuation mode.
    ctrl_ranges = list(
        zip(consts.ACTUATION_LIMITS[self._actuation]['min'],
            consts.ACTUATION_LIMITS[self._actuation]['max']))

    # Construct list of forcerange tuples from effort limits.
    force_ranges = list(
        zip(consts.EFFORT_LIMITS['min'], consts.EFFORT_LIMITS['max']))

    def add_actuator(i: int) -> types.MjcfElement:
      act_params = _PANDA_ACTUATOR_PARAMS[self._actuation][i]
      actuator = self._mjcf_root.actuator.add('general',
                                              name=f'j{i}',
                                              ctrllimited=True,
                                              forcelimited=True,
                                              ctrlrange=ctrl_ranges[i],
                                              forcerange=force_ranges[i],
                                              dyntype=dyntype,
                                              biastype='affine',
                                              gainprm=act_params.gainprm,
                                              biasprm=act_params.biasprm)
      actuator.joint = self._joints[i]
      return actuator

    self._actuators = [add_actuator(i) for i in range(consts.NUM_DOFS)]


class ExternalWrenchObserver(sensor.Sensor):
  """Estimates external wrench based on torque sensor signal."""
  _jac_pos: np.ndarray
  _jac_rot: np.ndarray
  _dof_indices: Sequence[int]
  _site_id: int

  def __init__(self, robot_params: params.RobotParams, arm: Panda,
               arm_sensor: robot_arm_sensor.RobotArmSensor) -> None:
    self._arm = arm
    self._name = robot_params.name
    self._arm_sensor = arm_sensor
    self._frame = robot_params.control_frame
    self._read_torques = arm_sensor.observables[self._arm_sensor.get_obs_key(
        robot_arm_sensor.joint_observations.Observations.JOINT_TORQUES)]
    self._observables = {
        self.get_obs_key(wrench_observations.Observations.FORCE):
            observable.Generic(self._force),
        self.get_obs_key(wrench_observations.Observations.TORQUE):
            observable.Generic(self._torque)
    }
    for obs in self._observables.values():
      obs.enabled = True

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  def after_compile(self, mjcf_model: mjcf.RootElement,
                    physics: mjcf.Physics) -> None:
    indexer = physics.named.model.dof_jntid.axes.row
    self._dof_indices = indexer.convert_key_item(
        [j.full_identifier for j in self._arm.joints])
    jac = np.empty((6, physics.model.nv))
    self._jac_pos, self._jac_rot = jac[:3], jac[3:]
    self._site_id = physics.model.name2id(self._arm.wrist_site.full_identifier,
                                          'site')

  @property
  def name(self) -> str:
    return self._name

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  def get_obs_key(self, obs: enum.Enum) -> str:
    return obs.get_obs_key(self.name)

  def _force(self, physics: mjcf.Physics) -> np.ndarray:
    mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, self._jac_pos,
                      self._jac_rot, self._site_id)
    f = self._jac_pos[:, self._dof_indices] @ self._read_torques(physics).copy()
    f = np.concatenate([f, np.zeros(3)])
    return geometry.WrenchStamped(f, None).get_relative_wrench(
        self._frame, mujoco_physics.wrap(physics)).force

  def _torque(self, physics: mjcf.Physics) -> np.ndarray:
    mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, self._jac_pos,
                      self._jac_rot, self._site_id)
    tau = self._jac_rot[:,
                        self._dof_indices] @ self._read_torques(physics).copy()
    tau = np.concatenate([np.zeros(3), tau])
    return geometry.WrenchStamped(tau, None).get_relative_wrench(
        self._frame, mujoco_physics.wrap(physics)).torque


class Cartesian6dVelocityEffector(
    cartesian_6d_velocity_effector.Cartesian6dVelocityEffector):
  """Panda Version of the MoMa Cartesian6dVelocityEffector."""

  def __init__(self, robot_params: params.RobotParams, arm: robot_arm.RobotArm,
               gripper: robot_hand.RobotHand,
               joint_velocity_effector: effector.Effector,
               tcp_sensor: robot_tcp_sensor.RobotTCPSensor):
    self._frame = robot_params.control_frame
    self._arm = arm
    self._get_world_pos = tcp_sensor.observables[tcp_sensor.get_obs_key(
        robot_tcp_sensor.Observations.POS)]
    model_params = cartesian_6d_velocity_effector.ModelParams(
        gripper.tool_center_point, arm.joints)
    control_params = cartesian_6d_velocity_effector.ControlParams(
        0.1,
        joint_position_limit_velocity_scale=.95,
        minimum_distance_from_joint_position_limit=.01,
        joint_velocity_limits=np.array(consts.VELOCITY_LIMITS['max']))
    super().__init__(robot_params.name, joint_velocity_effector, model_params,
                     control_params)

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    stamped_command = geometry.TwistStamped(command, self._frame)
    world_twist = stamped_command.get_world_twist(mujoco_physics.wrap(physics),
                                                  rot_only=True).full.copy()
    super().set_control(physics, world_twist)


class ArmEffector(arm_effector.ArmEffector):
  """Robot arm effector for the Panda MoMa model.
  
  Takes :py:class:`dm_robotics.panda.parameters.RobotParams`
  and changes the joint stiffness and damping of the robot arm. Otherwise behaves
  like :py:class:`dm_robotics.moma.effectors.arm_effector.ArmEffector`.
  """

  def __init__(self, robot_params: params.RobotParams, arm: robot_arm.RobotArm):
    """Constructor.

    Args:
      robot_params: Dataclass containing robot parameters.
      arm: The MoMa arm to control.
    """
    super().__init__(arm, None, robot_params.name)
    self._robot_params = robot_params
    self._empty_spec = specs.BoundedArray(shape=(0,),
                                          dtype=np.float32,
                                          minimum=0,
                                          maximum=0)

  def after_compile(self, mjcf_model: mjcf.RootElement,
                    physics: mjcf.Physics) -> None:
    if self._robot_params.actuation in [
        consts.Actuation.CARTESIAN_VELOCITY, consts.Actuation.JOINT_VELOCITY,
        consts.Actuation.HAPTIC
    ]:
      utils.set_joint_stiffness(self._robot_params.joint_stiffness, self._arm,
                                physics)
      utils.set_joint_damping(self._robot_params.joint_damping, self._arm,
                              physics)

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    if self._robot_params.actuation == consts.Actuation.HAPTIC:
      return
    super().set_control(physics, command)

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    if self._robot_params.actuation == consts.Actuation.HAPTIC:
      return self._empty_spec
    return super().action_spec(physics)


@enum.unique
class ControlObservations(enum.Enum):
  """Observations exposed by this sensor in control frame."""
  POS = '{}_pos_control'
  QUAT = '{}_quat_control'
  POSE = '{}_pose_control'
  RMAT = '{}_rmat_control'
  VEL = '{}_vel_control'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)


class RobotTCPSensor(site_sensor.SiteSensor):
  """Version of `dm_robotics.moma.sensors.site_sensor.SiteSensor` that
  takes tool center point pose measurements of a gripper and accepts
  `parameters.RobotParams` to optionally change the reference frame.
  Otherwise behaves like a `SiteSensor`.
  """

  def __init__(self, gripper: robot_hand.AnyRobotHand,
               robot_params: params.RobotParams):
    """Constructor.

    Args:
      gripper: This gripper's TCP site is used for the measurements.
        robot_params: Set the `control_frame` field to a site to use
        as reference frame. Falls back to world frame if `None`.
    """
    super().__init__(gripper.tool_center_point, f'{robot_params.name}_tcp')
    self._frame = robot_params.control_frame
    self._observables.update({
        self.get_obs_key(ControlObservations.POS):
            observable.Generic(self._pos_control),
        self.get_obs_key(ControlObservations.QUAT):
            observable.Generic(self._quat_control),
        self.get_obs_key(ControlObservations.RMAT):
            observable.Generic(self._rmat_control),
        self.get_obs_key(ControlObservations.POSE):
            observable.Generic(self._pose_control),
        self.get_obs_key(ControlObservations.VEL):
            observable.Generic(self._vel_control),
    })
    for obs in self._observables.values():
      obs.enabled = True

  def _pos_control(self, physics: mjcf.Physics) -> np.ndarray:
    return self._pose_control(physics)[:3]

  def _quat_control(self, physics: mjcf.Physics) -> np.ndarray:
    return self._pose_control(physics)[3:]

  def _pose_control(self, physics: mjcf.Physics) -> np.ndarray:
    pos = physics.bind(self._site).xpos
    quat = tr.mat_to_quat(np.reshape(physics.bind(self._site).xmat, [3, 3]))
    quat = tr.positive_leading_quat(quat)
    return geometry.PoseStamped(geometry.Pose(pos, quat)).get_relative_pose(
        self._frame, mujoco_physics.wrap(physics)).to_posquat()

  def _rmat_control(self, physics: mjcf.Physics) -> np.ndarray:
    return tr.quat_to_mat(self._quat_control(physics))[:3, :3].reshape((-1,))

  def _vel_control(self, physics: mjcf.Physics) -> np.ndarray:
    return geometry.TwistStamped(super()._site_vel_world(physics),
                                 None).get_relative_twist(
                                     self._frame,
                                     mujoco_physics.wrap(physics)).full


class RobotArmSensor(robot_arm_sensor.RobotArmSensor):
  """Behaves like :py:class:`dm_robotics.moma.sensors.robot_arm_sensor.RobotArmSensor`.
  
  Except that the joint torque signal does not include passive forces.
  This is done so as to model the external torque signal provided by the Panda robot.
  """

  def __init__(self, robot_params: params.RobotParams, arm: robot_arm.RobotArm):
    super().__init__(arm, robot_params.name, True)

  def _joint_torques(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(
        self._arm.joint_torque_sensors).sensordata[2::3] - physics.bind(
            self._arm.joints).qfrc_passive  # pytype: disable=attribute-error


def build_robot(robot_params: params.RobotParams) -> robot.Robot:
  """Builds a MoMa robot model of the Panda."""
  robot_sensors = []
  arm = Panda(actuation=robot_params.actuation, name=robot_params.name)
  arm_sensor = RobotArmSensor(robot_params, arm)

  ns_gripper = f'{robot_params.name}_gripper'
  if robot_params.has_hand:
    gripper = gripper_module.PandaHand(name=ns_gripper)
    panda_hand_sensor = gripper_module.PandaHandSensor(gripper, ns_gripper)
    robot_sensors.append(panda_hand_sensor)
    gripper_effector = gripper_module.PandaHandEffector(robot_params, gripper,
                                                        panda_hand_sensor)
  elif robot_params.gripper is not None:
    gripper = robot_params.gripper.model
    if robot_params.gripper.sensors is not None:
      robot_sensors.extend(robot_params.gripper.sensors)
    gripper_effector = robot_params.gripper.effector
  else:
    gripper = gripper_module.DummyHand(name=ns_gripper)
    gripper_effector = None

  tcp_sensor = RobotTCPSensor(gripper, robot_params)
  robot_sensors.extend([
      ExternalWrenchObserver(robot_params, arm, arm_sensor), tcp_sensor,
      arm_sensor
  ])
  robot_sensors.reverse()

  if robot_params.actuation in [
      consts.Actuation.JOINT_VELOCITY, consts.Actuation.HAPTIC
  ]:
    _arm_effector = ArmEffector(robot_params, arm)
  elif robot_params.actuation == consts.Actuation.CARTESIAN_VELOCITY:
    joint_velocity_effector = ArmEffector(robot_params, arm)
    _arm_effector = Cartesian6dVelocityEffector(robot_params, arm, gripper,
                                                joint_velocity_effector,
                                                tcp_sensor)

  robot.standard_compose(arm, gripper)
  moma_robot = robot.StandardRobot(arm=arm,
                                   arm_base_site_name=arm.base_site.name,
                                   gripper=gripper,
                                   robot_sensors=robot_sensors,
                                   arm_effector=_arm_effector,
                                   gripper_effector=gripper_effector)
  return moma_robot
