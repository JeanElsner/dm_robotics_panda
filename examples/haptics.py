"""
Demo of the haptic actuation mode.

The haptic actuation mode allows you to use a Panda robot
to physically interact with a simulation in real-time.
"""
import os

import dm_env
import numpy as np
from dm_control import mjcf
from dm_env import specs

from dm_robotics.panda import arm_constants, env_builder
from dm_robotics.panda import parameters as params
from dm_robotics.panda import utils


class Agent:
  """
  There is no action required in haptic mode, i.e. this
  agent does nothing.
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

  # Load environment from an MJCF file.
  XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'haptics.xml')
  arena = mjcf.from_file(XML_PATH)

  # Robot parameters include the robot's IP for HIL,
  # haptic actuation mode and joint damping.
  robot_params = params.RobotParams(robot_ip=args.robot_ip,
                                    actuation=arm_constants.Actuation.HAPTIC,
                                    joint_damping=np.zeros(7))
  # Environment parameters consist of the MJCF element defined above
  # as arena and a higher control timestep for smooth interaction.
  env_params = params.EnvirontmentParameters(arena=arena, control_timestep=0.01)
  panda_env_builder = env_builder.PandaEnvironmentBuilder(robot_params,
                                                          env_params=env_params)

  with panda_env_builder.build_task_environment() as env:
    # Print the full action, observation and reward specification.
    utils.full_spec(env)
    # Initialize the agent.
    agent = Agent(env.action_spec())
    # Visualize the simulation to confirm physical interaction.
    app = utils.ApplicationWithPlot()
    app.launch(env, policy=agent.step)
