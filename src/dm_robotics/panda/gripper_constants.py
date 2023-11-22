"""Constants used by the gripper module."""
import enum
import os

XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'panda',
                        'panda_hand.xml')
"""Path to the Panda hand mjcf"""

GRIPPER_SITE_NAME = 'TCP'

ACTUATOR_NAMES = ['panda_hand_actuator']

JOINT_NAMES = ['panda_finger_joint1', 'panda_finger_joint2']

SPEED_TOLERANCE = 0.003
CONSECUTIVE_SAMPLES = 10


@enum.unique
class STATES(enum.Enum):
  """Possible gripper states."""
  READY = 0
  """Gripper can accept a new command."""
  WAITING = 1
  """Gripper will not accept new command."""
