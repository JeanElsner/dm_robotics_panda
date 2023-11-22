"""Complete reinforcement-learning environment example."""
import dm_env
import numpy as np
from dm_control import mjcf
from dm_control.composer.variation import distributions, rotations
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import rewards, timestep_preprocessor
from dm_robotics.geometry import pose_distribution
from dm_robotics.manipulation.props.rgb_objects import rgb_object
from dm_robotics.moma import entity_initializer, prop
from dm_robotics.moma.sensors import camera_sensor, prop_pose_sensor

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
  """Simple ball prop that consists of a MuJoco sphere geom."""

  def _build(self, *args, **kwargs):
    del args, kwargs
    mjcf_root = mjcf.RootElement()
    # Props need to contain a body called prop_root
    body = mjcf_root.worldbody.add('body', name='prop_root')
    body.add('geom',
             type='sphere',
             size=[0.04],
             solref=[0.01, 0.5],
             mass=1,
             rgba=(1, 0, 0, 1))
    super()._build('ball', mjcf_root)


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

  # Use the robot added by PandaEnvironment to add a MuJoCo camera element to the gripper.
  panda_env.robots[robot_params.name].gripper.tool_center_point.parent.add(
      'camera',
      pos=(.1, 0, -.1),
      euler=(180, 0, -90),
      fovy=90,
      name='wrist_camera')

  # Uniform distribution of 6D poses within the given bounds.
  gripper_pose_dist = pose_distribution.UniformPoseDistribution(
      min_pose_bounds=np.array(
          [0.5, -0.3, 0.7, .75 * np.pi, -.25 * np.pi, -.25 * np.pi]),
      max_pose_bounds=np.array(
          [0.1, 0.3, 0.1, 1.25 * np.pi, .25 * np.pi / 2, .25 * np.pi]))
  # The pose initializer uses the robot arm's position_gripper function
  # to set the end-effector pose from the distribution above.
  initialize_arm = entity_initializer.PoseInitializer(
      panda_env.robots[robot_params.name].position_gripper,
      gripper_pose_dist.sample_pose)

  # ComputeReward is a timestep preprocessor that accepts a callable which computes
  # a scalar reward based on the observation and adds it to the timestep.
  # We configure the validation frequency so this reward is computed for every timestep.
  reward = rewards.ComputeReward(
      goal_reward,
      validation_frequency=timestep_preprocessor.ValidationFrequency.ALWAYS)

  # Instantiate props
  ball = Ball()
  props = [ball]
  for i in range(10):
    props.append(rgb_object.RandomRgbObjectProp(color=(.5, .5, .5, 1)))

  # Extra camera sensor that adds camera observations including rendered images.
  cam_sensor = camera_sensor.CameraImageSensor(
      panda_env.robots[robot_params.name].gripper.mjcf_model.find(
          'camera', 'wrist_camera'), camera_sensor.CameraConfig(has_depth=True),
      'wrist_cam')

  # Extra prop sensor to add ball pose to observation.
  goal_sensor = prop_pose_sensor.PropPoseSensor(ball, 'goal')

  # Props need to be added to the environment before instantiating the prop initializer.
  panda_env.add_props(props)

  # This prop initializer uses uniform random samples within the interval [-0.5, 0.5] for
  # positions and uniformly distributed random quaternion samples for orientation. The initializer
  # by default keeps sampling poses for props until a collision-free pose is found.
  initialize_props = entity_initializer.prop_initializer.PropPlacer(
      props,
      position=distributions.Uniform(-.5, .5),
      quaternion=rotations.UniformQuaternion())

  panda_env.add_timestep_preprocessors([reward])
  panda_env.add_entity_initializers([
      initialize_arm,
      initialize_props,
  ])

  # Extra sensors include a camera sensor as well as a pose sensor for the ball.
  panda_env.add_extra_sensors([cam_sensor, goal_sensor])

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
