from unittest.mock import MagicMock

import pytest

from dm_robotics.panda import utils


@pytest.fixture
def mock_env():
  return MagicMock()


@pytest.fixture
def mock_runtime():
  return MagicMock()


def test_full_spec(capfd, mock_env):
  # Set up expected outputs
  expected_action_spec = 'Mocked Action Spec'
  expected_obs_spec = {
      'obs_key1': 'Mocked Obs Spec 1',
      'obs_key2': 'Mocked Obs Spec 2'
  }
  expected_reward_spec = 'Mocked Reward Spec'

  # Mock the action_spec, observation_spec, and reward_spec methods
  mock_env.action_spec.return_value = expected_action_spec
  mock_env.observation_spec.return_value = expected_obs_spec
  mock_env.reward_spec.return_value = expected_reward_spec

  # Call the function with the mock environment
  utils.full_spec(mock_env)

  # Capture the printed output
  captured = capfd.readouterr()

  # Check if the printed output matches the expected output
  assert captured.out.strip().split(
      '\n')[0] == f'Action spec: {expected_action_spec}'
  assert captured.out.strip().split('\n')[1] == 'Observation spec:'
  assert captured.out.strip().split(
      '\n')[2] == f'\tobs_key1: {expected_obs_spec["obs_key1"]}'
  assert captured.out.strip().split(
      '\n')[3] == f'\tobs_key2: {expected_obs_spec["obs_key2"]}'
  assert captured.out.strip().split(
      '\n')[4] == f'Reward spec: {expected_reward_spec}'


def test_default_arg_parser():
  parser = utils.default_arg_parser()
  args = parser.parse_args(['-g'])
  assert args.gui == True


def test_plots(mock_runtime):
  obs = utils.ObservationPlot(mock_runtime)
  action = utils.ActionPlot(mock_runtime)
  reward = utils.RewardPlot(mock_runtime)
