"""Minimal working example of the dm_robotics Panda model."""
import dm_env
import numpy as np
from dm_env import specs

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """Agents are used to control a robot's actions given
  current observations and rewards. This agent does nothing.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Provides robot actions every control timestep."""
    del timestep  # not used
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Use RobotParams to customize Panda robots added to the environment.
  robot_params = params.RobotParams(robot_ip=args.robot_ip)
  panda_env = environment.PandaEnvironment(robot_params)

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
