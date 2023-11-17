""" Minimal working example of the dm_robotics Panda model. """
import typing

import dm_env
import numpy as np
from dm_control import composer, mjcf
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import (observation_transforms,
                                                 rewards, timestep_preprocessor)
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import entity_initializer, prop, robot, sensor
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.sensors import prop_pose_sensor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """ Agents are used to control a robot's actions given
  current observations and rewards. This agent does nothing.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """ Provides robot actions every control-timestep. """
    del timestep  # not used
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  class Ball(prop.Prop):

    def _build(self, *args, **kwargs):
      mjcf_root = mjcf.RootElement()
      body = mjcf_root.worldbody.add('body', name='prop_root')
      body.add('geom', type='sphere', size=[0.05], solref=[0.01, 0.5], mass=10)
      super()._build('ball', mjcf_root)

    def set_initializer_pose(self, physics: mjcf.Physics, pos: np.ndarray,
                             quat: np.ndarray,
                             random_state: np.random.RandomState):
      del random_state
      self.set_pose(physics, pos, quat)

  ball = Ball()
  ball_pose_dist = pose_distribution.UniformPoseDistribution(
      min_pose_bounds=[0.2, -0.5, 0.1, 0, 0, 0],
      max_pose_bounds=[0.5, 0.5, 0.5, 0, 0, 0])
  ball_initializer = entity_initializer.PoseInitializer(
      ball.set_initializer_pose, ball_pose_dist.sample_pose)

  def ball_reward(observation: spec_utils.ObservationValue):
    ball_distance = np.linalg.norm(observation['ball_pose'][:3] -
                                   observation['panda_tcp_pos'])
    return np.clip(1.0 - ball_distance, 0, 1)

  reward = rewards.ComputeReward(
      ball_reward,
      validation_frequency=timestep_preprocessor.ValidationFrequency.ALWAYS)

  # Use RobotParams to customize Panda robots added to the environment.
  robot_params = params.RobotParams(robot_ip=args.robot_ip)
  panda_env = environment.PandaEnvironment(robot_params)

  panda_env.add_timestep_preprocessors([reward])
  panda_env.add_props([ball])
  panda_env.add_entity_initializers([ball_initializer])
  panda_env.add_extra_sensors([prop_pose_sensor.PropPoseSensor(ball, 'ball')])

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    utils.full_spec(env)
    # Initialize the agent
    agent = Agent(env.action_spec())
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=1000, real_time=True)
