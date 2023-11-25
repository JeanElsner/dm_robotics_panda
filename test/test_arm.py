import numpy as np
from dm_control import mjcf
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.sensors import robotiq_gripper_sensor
from numpy import testing

from dm_robotics.panda import arm, arm_constants, parameters


def test_physics_step():
  robot = arm.Panda()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()


def test_set_joint_angles():
  robot = arm.Panda()
  robot_params = parameters.RobotParams()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()
  robot.set_joint_angles(physics, robot_params.joint_positions)
  testing.assert_allclose(robot_params.joint_positions,
                          physics.bind(robot.joints).qpos)


def test_arm_effector():
  robot = arm.Panda()
  robot_params = parameters.RobotParams()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  effector = arm.ArmEffector(robot_params, robot)
  effector.set_control(physics, np.zeros(7, dtype=np.float32))
  robot.after_substep(physics, np.random.RandomState(0))
  effector.close()


def test_arm_haptic():
  robot = arm.Panda()
  robot_params = parameters.RobotParams(
      actuation=arm_constants.Actuation.HAPTIC)
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  effector = arm.ArmEffector(robot_params, robot)
  effector.set_control(physics, np.zeros(0, dtype=np.float32))


def test_custom_gripper():
  gripper = robotiq_2f85.Robotiq2F85()
  gripper_params = parameters.GripperParams(
      model=gripper,
      effector=default_gripper_effector.DefaultGripperEffector(
          gripper, 'robotique'),
      sensors=[
          robotiq_gripper_sensor.RobotiqGripperSensor(gripper, 'robotique')
      ])
  arm.build_robot(parameters.RobotParams(gripper=gripper_params))
