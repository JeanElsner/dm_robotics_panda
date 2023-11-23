import numpy as np
from dm_control import mjcf
from numpy import testing

from dm_robotics.panda import gripper, parameters


def test_physics_step():
  robot = gripper.PandaHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()

  assert len(robot.joints) == 2
  assert len(robot.actuators) == 1

  robot = gripper.DummyHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()

  assert len(robot.joints) == 0
  assert len(robot.actuators) == 0


def test_set_width():
  robot = gripper.PandaHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()

  robot.set_width(physics, 0)
  testing.assert_allclose(physics.bind(robot.joints).qpos, np.zeros(2))

  robot.set_width(physics, 0.08)
  testing.assert_allclose(physics.bind(robot.joints).qpos, 0.04 * np.ones(2))


def test_effector():
  robot_params = parameters.RobotParams()
  robot = gripper.PandaHand()
  sensor = gripper.PandaHandSensor(robot, 'hand')
  effector = gripper.PandaHandEffector(robot_params, robot, sensor)
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  effector.set_control(physics, np.zeros(1))
