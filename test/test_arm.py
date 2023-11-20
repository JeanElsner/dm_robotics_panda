from dm_control import mjcf

from dm_robotics.panda import arm


def test_physics_step():
  robot = arm.Panda()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()
