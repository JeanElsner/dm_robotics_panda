"""MoMa effectors and sensors for the Panda hardware."""
import logging
import threading
import time

import numpy as np
import panda_py
from dm_control import mjcf
from dm_robotics.geometry import geometry, mujoco_physics
from dm_robotics.moma import robot
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.sensors import robot_arm_sensor
from panda_py import controllers, libfranka

from . import arm as arm_module
from . import arm_constants
from . import gripper as gripper_module
from . import parameters as params

log = logging.getLogger("hardware")


class ArmEffector(arm_module.ArmEffector):
  """Panda hardware version of the ArmEffector."""

  def __init__(self, robot_params: params.RobotParams, arm: robot_arm.RobotArm,
               hardware: panda_py.Panda):
    super().__init__(robot_params, arm)
    self.hardware = hardware
    self.init_hardware(robot_params)

  def init_hardware(self, robot_params: params.RobotParams) -> None:
    """Initialize the hardware.
    
    Initializes the necessary controllers based on actuation mode and
    moves the robot into the initial joint positions."""
    if not self.hardware.move_to_joint_position(robot_params.joint_positions):
      raise RuntimeError('Failed to reach initial robot joint positions.')
    self.actuation = robot_params.actuation
    if self.actuation in [
        arm_constants.Actuation.CARTESIAN_VELOCITY,
        arm_constants.Actuation.JOINT_VELOCITY
    ]:
      controller = controllers.IntegratedVelocity(
          stiffness=robot_params.joint_stiffness,
          damping=robot_params.joint_damping)
    if self.actuation == arm_constants.Actuation.HAPTIC:
      controller = controllers.AppliedTorque(damping=robot_params.joint_damping)
    self.controller = controller
    self.hardware.start_controller(self.controller)

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    """Send either a joint velocity or joint torque signal to the physical
    robot, depending on current actuation mode."""
    self.fdir()
    if self.actuation in [
        arm_constants.Actuation.CARTESIAN_VELOCITY,
        arm_constants.Actuation.JOINT_VELOCITY
    ]:
      self.controller.set_control(command)
    if self.actuation == arm_constants.Actuation.HAPTIC:
      self.controller.set_control(
          physics.bind(self._arm.joints).qfrc_constraint * .1)
      command = np.array(self.hardware.get_state().q, dtype=np.float32)
      super().set_control(physics, command)

  def close(self):
    self.hardware.stop_controller()

  def fdir(self) -> None:
    """Error detection and recovery."""
    try:
      self.hardware.raise_error()
    except RuntimeError as e:
      log.error(e)
      self.hardware.recover()
      self.hardware.start_controller(self.controller)


class RobotArmSensor(robot_arm_sensor.RobotArmSensor):
  """Panda hardware version of the MoMa RobotArmSensor."""

  def __init__(self, robot_params: params.RobotParams, arm: robot_arm.RobotArm,
               hardware: panda_py.Panda):
    self.hardware = hardware
    super().__init__(arm, robot_params.name, True)

  def close(self):
    self.hardware.stop_controller()

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    self._arm.set_joint_angles(physics, self._joint_pos(physics))

  def _joint_pos(self, physics: mjcf.Physics) -> np.ndarray:
    physics_actuators = physics.bind(self._arm.actuators)

    if self._arm.actuation in [
        arm_constants.Actuation.CARTESIAN_VELOCITY,
        arm_constants.Actuation.JOINT_VELOCITY
    ]:
      # physics_actuators.act[:] = self.hardware.get_state().q
      self._arm.set_joint_angles(physics, self.hardware.q)
    elif self._arm.actuation == arm_constants.Actuation.HAPTIC:
      physics_actuators.ctrl[:] = self.hardware.get_state().q
    return physics.bind(self._arm.joints).qpos

  def _joint_vel(self, physics: mjcf.Physics) -> np.ndarray:
    physics.bind(self._arm.joints).qvel[:] = self.hardware.get_state().dq
    return physics.bind(self._arm.joints).qvel

  def _joint_torques(self, physics: mjcf.Physics) -> np.ndarray:
    physics.bind(
        self._arm.joint_torque_sensors
    ).sensordata[2::3] = self.hardware.get_state().tau_ext_hat_filtered
    return physics.bind(self._arm.joint_torque_sensors).sensordata[2::3]


class ExternalWrenchObserver(arm_module.ExternalWrenchObserver):
  """Reads the Panda robot's estimate of external wrenches."""

  def __init__(self, robot_params: params.RobotParams, arm: arm_module.Panda,
               arm_sensor: RobotArmSensor, hardware: panda_py.Panda) -> None:
    super().__init__(robot_params, arm, arm_sensor)
    self.hardware = hardware

  def _force(self, physics: mjcf.Physics) -> np.ndarray:
    if len(self.hardware.get_state().O_F_ext_hat_K) != 6:
      return np.zeros(6)
    return geometry.WrenchStamped(self.hardware.get_state().O_F_ext_hat_K,
                                  self._arm.base_site).get_relative_wrench(
                                      self._frame,
                                      mujoco_physics.wrap(physics)).force

  def _torque(self, physics: mjcf.Physics) -> np.ndarray:
    if len(self.hardware.get_state().O_F_ext_hat_K) != 6:
      return np.zeros(6)
    return geometry.WrenchStamped(self.hardware.get_state().O_F_ext_hat_K,
                                  self._arm.base_site).get_relative_wrench(
                                      self._frame,
                                      mujoco_physics.wrap(physics)).torque

  def close(self):
    self.hardware.stop_controller()


class PandaHandSensor(gripper_module.PandaHandSensor):
  """Hardware version of PandaHandSensor."""

  def __init__(self, robot_params: params.RobotParams,
               gripper: gripper_module.PandaHand,
               hardware: libfranka.Gripper) -> None:
    super().__init__(gripper, f'{robot_params.name}_gripper')
    self.hardware = hardware
    self.__width = 0.08
    self._running = True
    self.thread = threading.Thread(target=self._observe)
    self.thread.start()

  def _observe(self):
    while self._running:
      s = self.hardware.read_once()
      self.__width = s.width
      time.sleep(0.1)

  def _width(self, physics: mjcf.Physics) -> np.ndarray:
    width = self.__width
    physics.bind(self._gripper.actuators).ctrl[:] = width / 0.08
    return np.array(width, dtype=np.float32)

  def close(self):
    self._running = False
    self.thread.join()


class PandaHandEffector(gripper_module.PandaHandEffector):
  """Hardware version of PandaHandEffector."""

  def __init__(self, robot_params: params.RobotParams,
               gripper: robot_hand.RobotHand,
               panda_hand_sensor: PandaHandSensor, hardware: libfranka.Gripper):
    super().__init__(robot_params, gripper, panda_hand_sensor)
    self.hardware = hardware
    self._command = None
    self._last_command = None

    self._running = True
    self._monitor_thread = threading.Thread(target=self._run)
    self._monitor_thread.start()

  def _run(self):
    try:
      self.hardware.stop()
    except RuntimeError:
      log.warning('Stopping gripper action failed.')
    while self._running:
      time.sleep(0.1)
      command = self._command
      last_command = self._last_command
      if command is not None and command != last_command:
        self._last_command = command
        self.hardware.grasp(command, 0.2, 20, 0.08, 0.08)

  def close(self):
    self._running = False
    self._monitor_thread.join()

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    self._command = 0 if command[0] < 0.5 else 0.08


def build_robot(robot_params: params.RobotParams) -> robot.Robot:
  """Builds a MoMa robot model of the Panda with hardware in the loop."""
  hardware_panda = panda_py.Panda(robot_params.robot_ip)
  hardware_panda.set_default_behavior()
  hardware_panda.get_robot().set_collision_behavior(
      [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
      [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
      [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
      [100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

  robot_sensors = []
  panda = arm_module.Panda(actuation=robot_params.actuation,
                           name=robot_params.name,
                           hardware=hardware_panda)
  arm_sensor = RobotArmSensor(robot_params, panda, hardware_panda)

  ns_gripper = f'{robot_params.name}_hand'
  if robot_params.has_hand:
    hardware_gripper = libfranka.Gripper(robot_params.robot_ip)
    gripper = gripper_module.PandaHand(name=ns_gripper)
    panda_hand_sensor = PandaHandSensor(robot_params, gripper, hardware_gripper)
    robot_sensors.append(panda_hand_sensor)
    gripper_effector = PandaHandEffector(robot_params, gripper,
                                         panda_hand_sensor, hardware_gripper)
  elif robot_params.gripper is not None:
    gripper = robot_params.gripper.model
    if robot_params.gripper.sensors is not None:
      robot_sensors.extend(robot_params.gripper.sensors)
    gripper_effector = robot_params.gripper.effector
  else:
    gripper = gripper_module.DummyHand(name=ns_gripper)
    gripper_effector = None

  tcp_sensor = arm_module.RobotTCPSensor(gripper, robot_params)
  robot_sensors.extend([
      ExternalWrenchObserver(robot_params, panda, arm_sensor, hardware_panda),
      tcp_sensor, arm_sensor
  ])

  if robot_params.actuation in [
      arm_constants.Actuation.JOINT_VELOCITY, arm_constants.Actuation.HAPTIC
  ]:
    _arm_effector = ArmEffector(robot_params, panda, hardware_panda)
  elif robot_params.actuation == arm_constants.Actuation.CARTESIAN_VELOCITY:
    joint_velocity_effector = ArmEffector(robot_params, panda, hardware_panda)
    _arm_effector = arm_module.Cartesian6dVelocityEffector(
        robot_params, panda, gripper, joint_velocity_effector, tcp_sensor)

  robot.standard_compose(panda, gripper)
  moma_robot = robot.StandardRobot(arm=panda,
                                   arm_base_site_name=panda.base_site.name,
                                   gripper=gripper,
                                   robot_sensors=robot_sensors,
                                   arm_effector=_arm_effector,
                                   gripper_effector=gripper_effector)
  return moma_robot
