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

.. youtube:: hn42udf0uKc


Multiple Robots
---------------

Rewards and Observations
------------------------

Domain Randomization
--------------------