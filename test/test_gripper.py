import numpy as np
from dm_control import mjcf
from numpy import testing

from dm_robotics.panda import gripper


def test_physics_step():
  robot = gripper.PandaHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()

  robot = gripper.DummyHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()


def test_set_width():
  robot = gripper.PandaHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()

  robot.set_width(physics, 0)
  testing.assert_allclose(physics.bind(robot.joints).qpos, np.zeros(2))

  robot.set_width(physics, 0.08)
  testing.assert_allclose(physics.bind(robot.joints).qpos, 0.04 * np.ones(2))
