"""Example showing how to use a custom gripper with the Panda model."""
import dm_env
import numpy as np
from dm_env import specs
from dm_robotics.moma.effectors import default_gripper_effector
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.sensors import robotiq_gripper_sensor

from dm_robotics.panda import environment
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils


class Agent:
  """Agents are used to control a robot's actions given
  current observations and rewards.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """Sinusoidal movement of the gripper's fingers."""
    time = timestep.observation['time'][0]
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    action[6] = 255 * np.sin(time)
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  gripper = robotiq_2f85.Robotiq2F85()
  gripper_params = params.GripperParams(
      model=gripper,
      effector=default_gripper_effector.DefaultGripperEffector(
          gripper, 'robotique'),
      sensors=[
          robotiq_gripper_sensor.RobotiqGripperSensor(gripper, 'robotique')
      ])

  # Use RobotParams to customize Panda robots added to the environment.
  robot_params = params.RobotParams(gripper=gripper_params, has_hand=False)
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
