from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dm_control import mjcf
from dm_robotics.moma.sensors import joint_observations, wrench_observations

from dm_robotics.panda import arm, hardware, parameters


def test_build():
  with patch('panda_py.Panda') as mock_panda:
    with patch('panda_py.libfranka.Gripper') as mock_gripper:
      robot_params = parameters.RobotParams()
      robot = hardware.build_robot(robot_params)
      robot.arm_effector.close()
      robot.gripper_effector.close()
      for s in robot.sensors:
        s.close()


@pytest.fixture
def mock_hardware():
  hw = MagicMock()
  hw.q = np.zeros(7)
  mock_state = MagicMock()
  mock_state.tau_ext_hat_filtered = np.zeros(7)
  mock_state.dq = np.zeros(7)
  hw.get_state.return_value = mock_state
  return hw


def test_arm_effector(mock_hardware):
  robot = arm.Panda()
  robot_params = parameters.RobotParams()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  effector = hardware.ArmEffector(robot_params, robot, mock_hardware)
  effector.set_control(physics, np.zeros(7, dtype=np.float32))


def test_robot_sensor(mock_hardware):
  robot = arm.Panda()
  robot_params = parameters.RobotParams()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  sensor = hardware.RobotArmSensor(robot_params, robot, mock_hardware)

  sensor.observables[sensor.get_obs_key(
      joint_observations.Observations.JOINT_POS)](physics)
  sensor.observables[sensor.get_obs_key(
      joint_observations.Observations.JOINT_VEL)](physics)
  sensor.observables[sensor.get_obs_key(
      joint_observations.Observations.JOINT_TORQUES)](physics)


def test_wrench_sensor(mock_hardware):
  robot = arm.Panda()
  robot_params = parameters.RobotParams()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  arm_sensor = hardware.RobotArmSensor(robot_params, robot, mock_hardware)
  sensor = hardware.ExternalWrenchObserver(robot_params, robot, arm_sensor,
                                           mock_hardware)
  sensor.observables[sensor.get_obs_key(
      wrench_observations.Observations.FORCE)](physics)
  sensor.observables[sensor.get_obs_key(
      wrench_observations.Observations.TORQUE)](physics)
