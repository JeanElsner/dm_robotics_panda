from unittest.mock import patch

from dm_robotics.panda import hardware, parameters


def test_build():
  with patch('panda_py.Panda') as mock_panda:
    with patch('panda_py.libfranka.Gripper') as mock_gripper:
      robot_params = parameters.RobotParams()
      robot = hardware.build_robot(robot_params)
      robot.arm_effector.close()
      robot.gripper_effector.close()
      for s in robot.sensors:
        s.close()
