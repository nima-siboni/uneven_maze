# testing the uneven maze
from uneven_maze import sample_terrain_function, UnevenMaze
import gymnasium as gym
import numpy as np

config = {
    'width': 10,
    'height': 10,
    'mountain_height': 1.,
    'start_position': [0, 0],
    'goal_position': [9, 0],
    'max_steps': 100,
    'cost_height': 0.1,
    'cost_height_max': 1.,
    'cost_step': 0.01,
    'terrain_function': sample_terrain_function
}


def test_init(config=config):
    """
    Test the initialization of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)
    assert isinstance(env, UnevenMaze)

    # Test the parameters
    assert env.width == config['width']
    assert env.height == config['height']
    assert env.mountain_height == config['mountain_height']
    assert env._terrain_function == config['terrain_function']
    assert env.cost_height == config['cost_height']
    assert env.cost_height_max == config['cost_height_max']
    assert env.cost_step == config['cost_step']
    assert env._start_position == config['start_position']
    assert env.goal_position == config['goal_position']
    assert env._max_steps == config['max_steps']
    assert env._current_step == 0
    assert env._current_position == config['start_position']

    # Test the action space
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4

    # Test the observation space
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (3,)
    assert np.all(env.observation_space.low == np.array([0, 0, 0]))
    assert np.all(env.observation_space.high == np.array([1, 10, 10]))


# test the reset function
def test_reset(config=config):
    """
    Test the reset function of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    observation, info = env.reset()

    # Test the observation
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (3,)
    assert np.all(env.observation_space.low == np.array([0, 0, 0]))


    # Test the info
    assert info == {}


def test_step(config=config):
    """
    Test the step function of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    # Test the step function
    for action in range(4):
        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)

        # Test the observation
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (3,)
        assert np.all(observation >= np.array([0, 0, 0]))
        assert np.all(observation <= np.array([1, 10, 10]))

        # Test the reward
        assert isinstance(reward, float)

        # Test the terminated flag
        assert isinstance(terminated, bool)

        # Test the truncated flag
        assert isinstance(truncated, bool)

        # Test the info
        assert info == {}


def test_the_termination_condition(config=config):
    """
    If the agent tales height number of consequent actions to go up the agent should get
    terminated equal to True.
    :param config:
    :return:
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    terminated = False
    truncated = False
    # Test the step function
    for _ in range(config["height"] + 1):
        # Take a step going up, i.e. action 0
        observation, reward, terminated, truncated, info = env.step(action=0)

    assert terminated
    assert not truncated


# assert truncation after max_steps
def test_the_truncation_condition(config=config):
    """
    If the agent should get truncated equal to True after max_steps, given that it has not
    reached the terminal state.
    :param config:
    :return:
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    terminated = False
    truncated = False
    # Test the step function
    for i in range(config["max_steps"]):
        # Take a step going up, i.e. action 0 and 1 alternatively
        observation, reward, terminated, truncated, info = env.step(action=i % 2)

    assert not terminated
    assert truncated


def test_reward_function(config=config):
    """
    Test the reward function of the environment.
    :param config: the configuration of the environment
    :return: None
    """
    # Define the environment
    env = UnevenMaze(config)

    # Reset the environment
    env.reset()

    # Test the step function
    for action in range(4):
        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)

        # Test the reward
        assert isinstance(reward, float)
        assert reward <= 0.

    env.reset()

    # Going up should be costlier than the step cost
    _, r, _, _, _ = env.step(0)
    assert r < config["cost_step"]

    # Going down should be only as costly as the step cost
    _, r, _, _, _ = env.step(1)

    assert r == -1. * config["cost_step"]

    # Bumping your head to the wall should be as costly as the step cost
    _, r, _, _, _ = env.step(3)
    assert r == -1. * config["cost_step"]
