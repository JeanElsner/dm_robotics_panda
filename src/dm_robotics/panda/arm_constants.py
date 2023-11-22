"""Defines Panda robot arm constants."""

import enum
import os

XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'panda',
                        'panda.xml')
"""Path to the Panda arm mjcf"""


class Actuation(enum.Enum):
  """Available actuation methods for the Panda MoMa model.

  The actuation methods use the joint stiffness and damping as defined in
  :py:class:`dm_robotics.panda.parameters.RobotParams` where applicable.
  """
  CARTESIAN_VELOCITY = 0
  """Cartesian end-effector velocity control."""
  JOINT_VELOCITY = 1
  """Joint actuators receive a velocity signal that is integrated and
  tracked by a position controller."""
  HAPTIC = 2
  """Enables haptic interaction between a physical Panda robot and the simulation."""


# Number of degrees of freedom of the Panda robot arm.
NUM_DOFS = 7

# Joint names of Panda robot (without any namespacing). These names are defined
# by the Panda controller and are immutable.
JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]
WRIST_SITE_NAME = 'wrist_site'
BASE_SITE_NAME = 'panda_base'
JOINT_TORQUE_SENSOR_NAMES = [f'panda_joint_torque{i}' for i in range(1, 8)]

# Effort limits of the Panda robot arm in Nm.
EFFORT_LIMITS = {
    'min': (-87, -87, -87, -87, -12, -12, -12),
    'max': (87, 87, 87, 87, 12, 12, 12),
}

JOINT_LIMITS = {
    'min': (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    'max': (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
}

VELOCITY_LIMITS = {
    'min': (-1.74, -1.328, -1.957, -1.957, -3.485, -3.485, -4.545),
    'max': (1.74, 1.328, 1.957, 1.957, 3.485, 3.485, 4.545),
}

# Quaternion to align the attachment site with real
ROTATION_QUATERNION_MINUS_45DEG_AROUND_Z = (0.92387953, 0, 0, -0.38268343)

# Actuation limits of the Sawyer robot arm.
ACTUATION_LIMITS = {
    Actuation.CARTESIAN_VELOCITY: VELOCITY_LIMITS,
    Actuation.JOINT_VELOCITY: VELOCITY_LIMITS,
    Actuation.HAPTIC: JOINT_LIMITS,
}
