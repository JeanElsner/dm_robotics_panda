from dm_control import mjcf

from dm_robotics.panda import gripper


def test_physics_step():
  robot = gripper.PandaHand()
  physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
  physics.step()
