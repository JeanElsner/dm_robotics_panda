"""
Produces a Cartesian motion using the Cartesian actuation mode.
"""
import math

import dm_env
import numpy as np
from dm_env import specs

from dm_robotics.panda import env_builder
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """
  The agent produces a trajectory tracing the path of an eight
  in the x/y frame of the robot using end-effector velocities.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Computes velocities in the x/y plane parameterized in time.
    """
    time = timestep.observation['time'][0]
    r = 0.1
    vel_x = r * math.cos(time)  # Derivative of x = sin(t)
    vel_y = r * ((math.cos(time) * math.cos(time)) -
                 (math.sin(time) * math.sin(time)))
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    # The action space of the Cartesian 6D effector corresponds
    # to the linear and angular velocities in x, y and z directions
    # respectively
    action[0] = vel_x
    action[1] = vel_y
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # The Panda model and environment support many customization
  # parameters. Here we use only the defaults and robot IP if provided.
  robot_params = params.RobotParams(robot_ip=args.robot_ip)
  panda_env_builder = env_builder.PandaEnvironmentBuilder(robot_params)

  with panda_env_builder.build_task_environment() as env:
    # Print the full action, observation and reward specification.
    utils.full_spec(env)
    # Initialize the agent.
    agent = Agent(env.action_spec())
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=1000)
