"""Demo of the haptic actuation mode.

The haptic actuation mode allows you to use a Panda robot
to physically interact with a simulation in real-time.
"""
import os

import dm_env
import numpy as np
from dm_control import composer, mjcf
from dm_env import specs
from dm_control.viewer import user_input, application

from dm_robotics.panda import arm_constants, environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import utils


class Agent:
  """There is no action required in haptic mode, i.e. this agent does nothing."""

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Provides robot actions every control timestep."""
    del timestep  # not used
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    return action


class Obs(utils.ObservationPlot):
  """Selects torque observation to render by default."""
  def _init_buffer(self):
    super()._init_buffer()
    self._obs_idx = self._obs_keys.index('panda_torque')
    self.reset_data()
    self.update_title()


class App(utils.ApplicationWithPlot):
  """Sets default window size, camera and observation plot."""
  def __init__(self):
    super().__init__('Haptic Demo', 1920, 1080)

  def _perform_deferred_reload(self, params):
    application.Application._perform_deferred_reload(self, params)
    cmp = Obs(self._runtime)
    self._renderer.components += cmp
    self._renderer.components += utils.ActionPlot(self._runtime)
    self._renderer.components += utils.RewardPlot(self._runtime)
    self._input_map.bind(cmp.next_obs, user_input.KEY_F4)
    self._input_map.bind(cmp.prev_obs, user_input.KEY_F3)
    self._viewer._camera_select.select_next()


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Load environment from an MJCF file.
  XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'haptics.xml')
  arena = composer.Arena(xml_path=XML_PATH)

  # Robot parameters include the robot's IP for HIL,
  # haptic actuation mode and joint damping.
  robot_params = params.RobotParams(robot_ip=args.robot_ip,
                                    actuation=arm_constants.Actuation.HAPTIC,
                                    joint_damping=np.zeros(7))
  panda_env = environment.PandaEnvironment(robot_params,
                                           arena,
                                           control_timestep=0.01)

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification.
    utils.full_spec(env)
    # Initialize the agent.
    agent = Agent(env.action_spec())
    # Visualize the simulation to confirm physical interaction.
    app = App()
    app.launch(env, policy=agent.step)
