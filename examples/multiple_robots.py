"""Minimal working example demonstrating how to simulate multiple robots."""
import dm_env
import numpy as np
from dm_env import specs

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """Agents are used to control the robots' actions given
  current observations and rewards. This agent does nothing.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Provides robot actions every control-timestep."""
    del timestep  # not used
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Creating multiple robots in a simulation environment is easy,
  # simply create multiple robot parameters and turn them over
  # to the environment builder.
  robot_1 = params.RobotParams(name='robot_1', pose=[0, 0, 0, 0, 0, 0])
  robot_2 = params.RobotParams(name='robot_2',
                               pose=[.5, -.5, 0, 0, 0, np.pi * 3 / 4])
  robot_3 = params.RobotParams(name='robot_3',
                               pose=[.5, .5, 0, 0, 0, np.pi * 5 / 4])
  panda_env = environment.PandaEnvironment([robot_1, robot_2, robot_3])

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
