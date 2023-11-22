from dm_robotics.panda import environment, parameters


def test_robots():
  robot_config = [
      parameters.RobotParams(name='test1'),
      parameters.RobotParams(name='test2'),
      parameters.RobotParams(name='test3')
  ]
  panda_environment = environment.PandaEnvironment(robot_config)

  assert list(panda_environment.robots.keys())[0] == 'test1'
  assert len(panda_environment.robots) == 3


def test_build():
  with environment.PandaEnvironment(
      parameters.RobotParams()).build_task_environment():
    pass
