import numpy as np
from dm_control import mjcf
from dm_control.composer.variation import distributions, rotations
from dm_env import specs
from dm_robotics.agentflow.preprocessors import rewards
from dm_robotics.moma import effector, entity_initializer, prop
from dm_robotics.moma.sensors import prop_pose_sensor

from dm_robotics.panda import environment, parameters, utils


def test_robots():
  robot_config = [
      parameters.RobotParams(name='test1'),
      parameters.RobotParams(name='test2', pose=[0, 1, 0, 0, 0, 0]),
      parameters.RobotParams(name='test3', pose=[0, 2, 0, 0, 0, 0])
  ]
  panda_environment = environment.PandaEnvironment(robot_config)

  assert list(panda_environment.robots.keys())[0] == 'test1'
  assert len(panda_environment.robots) == 3


def test_build():
  with environment.PandaEnvironment(
      parameters.RobotParams()).build_task_environment() as env:
    utils.full_spec(env)


class MockEffector(effector.Effector):

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass

  def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
    pass

  @property
  def prefix(self) -> str:
    return 'dummy'

  def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
    return specs.BoundedArray((1,), np.float32, (0,), (1,))


def initialize_scene(random_state: np.random.RandomState) -> None:
  del random_state


def test_components():
  panda_env = environment.PandaEnvironment(parameters.RobotParams())

  props = [prop.Block()]
  extra_sensors = [prop_pose_sensor.PropPoseSensor(props[0], 'prop')]
  extra_effectors = [MockEffector()]
  preprocessors = [rewards.ComputeReward(lambda obs: 1)]

  panda_env.add_props(props)

  entity_initializers = [
      entity_initializer.prop_initializer.PropPlacer(
          props, distributions.Uniform(-1, 1), rotations.UniformQuaternion())
  ]
  panda_env.add_extra_effectors(extra_effectors)
  panda_env.add_extra_sensors(extra_sensors)
  panda_env.add_entity_initializers(entity_initializers)
  panda_env.add_scene_initializers([initialize_scene])
  panda_env.add_timestep_preprocessors(preprocessors)

  with panda_env.build_task_environment() as env:
    pass
