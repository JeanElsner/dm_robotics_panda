"""Two-arm robot example with local control frame."""
import os

import dm_env
import numpy as np
from dm_control import composer, mjcf
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
    """Provides robot actions every control-timestep."""
    time = timestep.observation['time'][0]
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    action[0] = 0.1 * np.sin(time)
    action[7] = action[0]
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Load environment from an MJCF file.
  # The environment includes a simple robot frame and
  # reference frame for our two-arm robot.
  XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'two_arm.xml')
  arena = composer.Arena(xml_path=XML_PATH)
  left_frame = arena.mjcf_model.find('site', 'left')
  right_frame = arena.mjcf_model.find('site', 'right')
  control_frame = arena.mjcf_model.find('site', 'control')

  # We use the sites defined in the MJCF to attach the robot arms
  # to the body. The robot's control site is used as a reference frame
  # for the arms' sensors and controllers.
  left = params.RobotParams(attach_site=left_frame,
                            name='left',
                            control_frame=control_frame)
  right = params.RobotParams(attach_site=right_frame,
                             name='right',
                             control_frame=control_frame)
  env_params = params.EnvirontmentParameters(mjcf_root=arena)
  panda_env = environment.PandaEnvironment([left, right], arena)

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
