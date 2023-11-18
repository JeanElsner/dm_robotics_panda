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

Joint
^^^^^

.. youtube:: C14HlT1Scdo

Cartesian
^^^^^^^^^

.. youtube:: vYvdr7iGCv4


Gripper
^^^^^^^

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