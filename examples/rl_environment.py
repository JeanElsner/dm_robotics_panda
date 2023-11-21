""" Minimal working example of the dm_robotics Panda model. """
import typing

import dm_env
import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.variation import distributions
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import (observation_transforms,
                                                 rewards, timestep_preprocessor)
from dm_robotics.geometry import pose_distribution
from dm_robotics.manipulation.props.rgb_objects import rgb_object
from dm_robotics.moma import entity_initializer, prop, robot, sensor
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.sensors import prop_pose_sensor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """Reactive agent that follows a goal."""

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Computes end-effector velocities in direction of goal."""
    observation = timestep.observation
    v = observation['goal_pose'][:3] - observation['panda_tcp_pos']
    # v = 0.1 * v / np.linalg.norm(v)
    v = max(np.linalg.norm(v), 0.1) * v / np.linalg.norm(v)
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    action[:3] = v
    action[6] = 1
    return action


class Ball(prop.Prop):

  def _build(self, *args, **kwargs):
    mjcf_root = mjcf.RootElement()
    body = mjcf_root.worldbody.add('body', name='prop_root')
    body.add('geom',
             type='sphere',
             size=[0.04],
             solref=[0.01, 0.5],
             mass=1,
             rgba=(1, 0, 0, 1))
    super()._build('goal', mjcf_root)

  def set_pose(self,
               physics: mjcf.Physics,
               position: np.ndarray,
               quaternion: np.ndarray,
               random_state: typing.Optional[np.random.RandomState] = None):
    del random_state
    super().set_pose(physics, position, quaternion)


def goal_reward(observation: spec_utils.ObservationValue):
  """Computes a normalized reward based on distance between end-effector and goal."""
  goal_distance = np.linalg.norm(observation['goal_pose'][:3] -
                                 observation['panda_tcp_pos'])
  return np.clip(1.0 - goal_distance, 0, 1)


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Use RobotParams to customize Panda robots added to the environment.
  robot_params = params.RobotParams(robot_ip=args.robot_ip)
  panda_env = environment.PandaEnvironment(robot_params)

  gripper_pose_dist = pose_distribution.UniformPoseDistribution(
      min_pose_bounds=np.array(
          [0.5, -0.3, 0.7, .75 * np.pi, -.25 * np.pi, -.25 * np.pi]),
      max_pose_bounds=np.array(
          [0.1, 0.3, 0.1, 1.25 * np.pi, .25 * np.pi / 2, .25 * np.pi]))
  initialize_arm = entity_initializer.PoseInitializer(
      panda_env.robots[robot_params.name].position_gripper,
      gripper_pose_dist.sample_pose)

  reward = rewards.ComputeReward(
      goal_reward,
      validation_frequency=timestep_preprocessor.ValidationFrequency.ALWAYS)

  ball = Ball()
  props = [ball]
  for i in range(5):
    props.append(rgb_object.RandomRgbObjectProp(color=(.5, .5, .5, 1)))

  initialize_props = entity_initializer.prop_initializer.PropPlacer(
      props, distributions.Uniform(-.5, .5))

  panda_env.add_timestep_preprocessors([reward])
  panda_env.add_props(props)
  panda_env.add_entity_initializers([
      initialize_arm,
      initialize_props,
  ])
  panda_env.add_extra_sensors([prop_pose_sensor.PropPoseSensor(ball, 'goal')])

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
