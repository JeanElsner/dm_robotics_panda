import logging
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest

from dm_robotics.panda import utils


@pytest.fixture
def mock_env():
  return MagicMock()


def test_default_arg_parser():
  parser = utils.default_arg_parser()
  args = parser.parse_args(['-g'])
  assert args.gui == True


@pytest.fixture
def mock_runtime():
  rt = MagicMock()
  # rt._time_step.observation.items.return_value = [('key1', np.zeros(1))]
  rt._time_step.observation = {'key1': np.zeros(1)}
  rt._time_step.reward = 0
  return rt


@pytest.fixture
def mock_context():
  return MagicMock()


@pytest.fixture
def mock_viewport():
  return MagicMock()


def test_plots(mock_runtime, mock_context, mock_viewport):
  with patch('mujoco.mjr_figure') as mock_mjr_figure:
    obs = utils.ObservationPlot(mock_runtime)
    obs.render(mock_context, mock_viewport)
    obs.next_obs()
    obs.prev_obs()
    action = utils.ActionPlot(mock_runtime)
    action.render(mock_context, mock_viewport)
    reward = utils.RewardPlot(mock_runtime)
    reward.render(mock_context, mock_viewport)


def test_logging():
  utils.init_logging()


def test_formatter_warning():
  formatter = utils.Formatter()

  record = logging.LogRecord(name='test_logger',
                             level=logging.WARNING,
                             pathname='/path/to/module.py',
                             lineno=42,
                             msg='This is a warning message',
                             args=(),
                             exc_info=None)

  formatted_msg = formatter.format(record)

  # Check if the formatted message contains the ANSI escape code for yellow color
  assert '\033[33m' in formatted_msg
  # Check if the formatted message ends with the ANSI escape code for resetting color
  assert formatted_msg.endswith('\033[0m')


def test_formatter_error():
  formatter = utils.Formatter()

  record = logging.LogRecord(name='test_logger',
                             level=logging.ERROR,
                             pathname='/path/to/module.py',
                             lineno=42,
                             msg='This is an error message',
                             args=(),
                             exc_info=None)

  formatted_msg = formatter.format(record)

  # Check if the formatted message contains the ANSI escape code for red color
  assert '\033[31m' in formatted_msg
  # Check if the formatted message ends with the ANSI escape code for resetting color
  assert formatted_msg.endswith('\033[0m')


def test_formatter_info():
  formatter = utils.Formatter()

  record = logging.LogRecord(name='test_logger',
                             level=logging.INFO,
                             pathname='/path/to/module.py',
                             lineno=42,
                             msg='This is an info message',
                             args=(),
                             exc_info=None)

  formatted_msg = formatter.format(record)

  assert '\033[33m' not in formatted_msg
  assert '\033[31m' not in formatted_msg
