"""
Produces a joint motion using the joint velocity actuation mode.
"""
import dm_env
import numpy as np
from dm_env import specs

from dm_robotics.panda import arm_constants, environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """
  This agent produces a sinusoidal joint movement.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Computes sinusoidal joint velocities.
    """
    time = timestep.observation['time'][0]
    action = 0.1 * np.sin(
        np.ones(shape=self._spec.shape, dtype=self._spec.dtype) * time)
    action[7] = 0  # gripper action
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # The Panda model and environment support many customization
  # parameters. Here we use only the robot IP if provided and set
  # the actuation mode to joint velocities.
  robot_params = params.RobotParams(
      robot_ip=args.robot_ip, actuation=arm_constants.Actuation.JOINT_VELOCITY)
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
