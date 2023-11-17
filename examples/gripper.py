"""
Example demonstrating use of the Panda robot's gripper.

The gripper's action space accepts one floating point number
between 0 and 1 but internally maps those values to 0 if < 0.5 and 1 otherwise,
where 0 corresponds to an inner and 1 to an outer grasp. The gripper's grasp
will adjust to an object's width and apply a continuous force either inwards
or outwards. The observed hysteresis of the gripper means that it will ignore
new commands until the the current grasp finished. This stems from the fact
the the real hardware cannot be controlled in real-time.
"""
import dm_env
import numpy as np
from dm_env import specs

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """
  This agent controls the gripper with random actions.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Every timestep, a new random gripper action is generated
    that would result in either an outward or inward grasp.
    However, the gripper only moves if 1) it is not already moving
    and 2) the new command is different from the last.
    Therefore this agent will effectively result in continuously
    opening and closing the gripper as quickly as possible.
    """
    del timestep
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    action[6] = np.random.rand()
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
  panda_env = environment.PandaEnvironment(robot_params)

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification.
    utils.full_spec(env)
    # Initialize the agent.
    agent = Agent(env.action_spec())
    # Run the environment and agent either in headless mode or inside the GUI.
    if args.gui:
      app = utils.ApplicationWithPlot()
      app.launch(env, policy=agent.step)
    else:
      run_loop.run(env, agent, [], max_steps=1000, real_time=True)
