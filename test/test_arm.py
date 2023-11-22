from dm_control import mjcf
from numpy import testing

from dm_robotics.panda import arm, parameters


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
