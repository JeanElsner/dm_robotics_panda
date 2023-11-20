Tutorial
========

Introduction
------------

This tutorial focuses on functionality provided by ``dm_robotics_panda``.
Although the examples provide a fully implemented starting point, for more details
on how to prepare reinforcement learning environments please refer to the material
provided with the `dm_robotics <https://github.com/google-deepmind/dm_robotics>`_
and by extension `dm_control <https://github.com/google-deepmind/dm_control>`_ repositories.

.. note::
    This section is implemented in ``examples/minimal_working_example.py``.
    Run this file or any of the examples with the ``--gui`` option or ``-g``
    to run the simulation in the visualization app.

A minimum working example consists of a robot configuration, an environment,
an agent that controls the robot, and a method to run the environment.
:py:class:`dm_robotics.panda.environment.PandaEnvironment` is the main point of interaction.
This class populates the environment with Panda robots configured using
:py:class:`dm_robotics.panda.parameters.RobotParams`.

.. code:: python

   robot_params = params.RobotParams()
   panda_env = environment.PandaEnvironment(robot_params)

   with panda_env.build_task_environment() as env:
     ...

The MoMa task environment returned by ``build_task_environment()`` can then be executed
by :py:func:`dm_robotics.panda.run_loop.run` or visualized with 
:py:class:`dm_robotics.panda.utils.ApplicationWithPlot`. The visualization app
is a convenient tool to debug environments or evaluate agents and
includes live plots of observations, actions, and rewards.

.. image:: img/gui.png
   :alt: Visualization App

Finally, we require an agent that provides actions in the form of NumPy arrays
according to the environment's specification. A minimal agent provides a ``step``
function that receives a timestep and returns an action.

.. code:: python

   class Agent:

     def __init__(self, spec: specs.BoundedArray) -> None:
       self._spec = spec

     def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
       action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
       return action

The shape of the action will depend on the number of configured robots and chosen actuation mode.


Motion Control
--------------

By default the robots use a Cartesian velocity motion
`controller <https://github.com/google-deepmind/dm_robotics/blob/main/cpp/controllers/README.md>`_
from ``dm_robotics`` that uses quadratic programming to solve a stack-of-tasks optimization problem.
The actuation mode is configured as part of the robot parameters as defined in
:py:class:`dm_robotics.panda.arm_constants.Actuation`.
The motion of the Panda robot is controlled through the agent interface.
Agents need to provide a ``step()`` function that accepts a ``dm_env`` timestep and returns
an action (control signal) in the form of a NumPy array.
 

.. note::
    The examples in this section support optional hardware in the loop (HIL) mode. You can run the examples
    with HIL by executing the files with the ``--robot-ip`` option set to the hostname or IP address
    of a robot connected to the host computer. This option can be combined with ``--gui`` for visualization.

    When running any of the examples
    the action specification (shape) of the configured actuation mode along with the observation
    and reward specification will be printed in the terminal for convenience.

Joint
^^^^^

Joint velocity control is activated as part of the robot configuration.

.. code:: python

   robot_params = params.RobotParams(actuation=arm_constants.Actuation.JOINT_VELOCITY)

The action interface is a 7-vector where each component controls the corresponding joint's velocity.
If the Panda gripper is used (default behavior) there is one editional component to control grasping.

.. code:: python

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


   agent = Agent(env.action_spec())

Where ``env.action_spec()`` is the MoMa subtask environment returned by ``build_task_environment()``
that is used to retrieve the environment's action specification. This example will result in a small
periodic motion and is implemented in ``examples/motion_joint.py``. See below for a video of the example
running with HIL and the visualizaiton app.

.. youtube:: C14HlT1Scdo

Cartesian
^^^^^^^^^

Cartesian velocity control is the default behavior but can also be configured explicitly
as part of a robot's configuration.

.. code:: python

   robot_params = params.RobotParams(actuation=arm_constants.Actuation.CARTESIAN_VELOCITY)

The effector's (controller's) action space consists of a 6-vector where the first three indices
correspond to the desired end-effector velocity along the control frame's x-, y-, and z-axis. The
latter three indices define the angular velocities respectively. If no control frame is configured,
the world frame is used as a reference.

.. todo::

   Refer to a section explaining control frames and their use in detail.


.. code:: python

   class Agent:
     """
     The agent produces a trajectory tracing the path of an eight
     in the x/y control frame of the robot using end-effector velocities.
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

This agent computes velocities parameterized in simulation time to produce a path
that roughly traces the shape of an eight in the x/y plane.
Note that this agent does not implement a trajectory follower but rather applies
end-effector velocities in an open loop manner. As such the path may drift over time.
In practice we would expect more sophisticated (learned) agents to take the current
observation (state) into account.

The video below demonstrates the example implemented in ``examples/motion_cartesian.py``
with HIL and visualization.

.. youtube:: vYvdr7iGCv4


Gripper
^^^^^^^

The Panda's gripper (officially called the Franka Hand) is not easy to model as it doesn't feature a
real-time control interface and is affected by hysteresis. Because of this, the gripper's MoMa effector
features only a binary action space, allowing for an outward and an inner grasp corresponding to action
values 0 and 1 respectively. Internally the gripper's effector maps actions to 0 if < 0.5 and 1 otherwise.
The gripper is attached by default, however this behavior can be deactivated or explicitly set in the
robot configuration.

.. code:: python

   robot_params = params.RobotParams(has_hand=True)

The example implemented in ``examples/gripper.py`` includes a simple agent that generates random actions
to illustrate the gripper's behavior.

.. code:: python

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

Note how, in the video below, the grasp adapts to the size of an object placed between
the gripper's fingers. This can also be observed in the gripper's width observation plot.

.. youtube:: h3P0HBPF3NU


Haptic Interaction
------------------

The haptic actuation mode is a special mode that renders constraint forces from
the simulation on the real robot when used with HIL. This allows users to haptically
interact with the simulation through the robot. Haptic mode is activated through the
robot configuration. Additional settings include ``joint_damping`` which is usually
set to 0.

.. code:: python

   robot_params = params.RobotParams(robot_ip=args.robot_ip,
                                     actuation=arm_constants.Actuation.HAPTIC,
                                     joint_damping=np.zeros(7))
   panda_env = environment.PandaEnvironment(robot_params,
                                            arena,
                                            control_timestep=0.01)

Setting the MoMa control timestep to a small value will improve quality and stability of the
physical interaction. The example implemented in ``examples/haptics.py`` loads a simple scene
from an MJCF file that includes a static cube.

.. code:: python

   # Load environment from an MJCF file.
   XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'haptics.xml')
   arena = composer.Arena(xml_path=XML_PATH)

The video below demonstrates haptic interaction mode. Note that the HIL connection feeds
measured external forces back into the simulation which can be accessed as observations.

.. youtube:: hn42udf0uKc


Multiple Robots
---------------

Populating an environment with multiple Panda robots is done by simply
creating multiple robot configurations and using them to instantiate
:py:class:`dm_robotics.panda.environment.PandaEnvironment`.

.. code:: python

   robot_1 = params.RobotParams(name='robot_1', pose=[0, 0, 0, 0, 0, 0])
   robot_2 = params.RobotParams(name='robot_2',
                                pose=[.5, -.5, 0, 0, 0, np.pi * 3 / 4])
   robot_3 = params.RobotParams(name='robot_3',
                                pose=[.5, .5, 0, 0, 0, np.pi * 5 / 4])
   panda_env = environment.PandaEnvironment([robot_1, robot_2, robot_3])

The ``pose`` parameter is a 6-vector that describes a transform (linear displacement and Euler angles)
to a robot's base. Without this configuration the three robots would spawn in same location.
A minimal example of multiple robot is implemented in ``examples/multiple_robots.py``
and shown below.

.. image:: img/multiple_robots.png
   :alt: Multiple Robots

In a more sophisticated application we can build a simple robot with a branching kinematic structure.
For this purpose we modeled a stationary robot with a hinge joint around its main axis. The MJCF file
includes ``site`` elements as part of the robot's body that define the attachment as well as control
frames.

.. code:: python

   XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'two_arm.xml')
   arena = composer.Arena(xml_path=XML_PATH)
   left_frame = arena.mjcf_model.find('site', 'left')
   right_frame = arena.mjcf_model.find('site', 'right')
   control_frame = arena.mjcf_model.find('site', 'control')

Using an attachment frame is different from using the ``pose`` parameter in so far as the
attached Panda robot will be a child of the ``body`` containing the attachment site. The
control frame is the reference frame used by the Cartesian velocity motion controller.
It is also used to compute pose, velocity and wrench observations in control frame.

.. code:: python

   left = params.RobotParams(attach_site=left_frame,
                             name='left',
                             control_frame=control_frame)
   right = params.RobotParams(attach_site=right_frame,
                              name='right',
                              control_frame=control_frame)
   env_params = params.EnvirontmentParameters(mjcf_root=arena)
   panda_env = environment.PandaEnvironment([left, right], arena)

By using the same control frame attached to the robot's body for both arms, the Cartesian motion of the
arms is invariant to the rotation of the robot (i.e. the reference frame rotates with the main body).
This can be seen in ``examples/two_arm_robot.py``. In this example the agent produces a sinusoidal velocity
action along the x-axis for both arms. In the video below, the user can be seen to interact with the robot
body to rotate it, while the motion remains invariant.

.. youtube:: cAUjkhrBmN4


Reward and Observation
----------------------


Domain Randomization
--------------------
