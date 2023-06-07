# Uneven_maze: This is a RL environment compatible with Gymnasium which represents a grid map (or
# maze) which is
# not flat. As a consequence the cost of taking one step depends on whether it is uphill or downhill.
# The agent is rewarded for reaching the goal and penalized for taking steps. The cost of the
# steps is a weighted sum of a constant step cost and the height difference between the start and
# end of the step. The weight is a parameter of the environment.
# The height in of the map is represented by a function of x, y coordinates. The function is
# specified as a parameter of the environment.
import copy
from typing import Callable, Dict, List, Tuple

# The parameters of the environment are:
# - the size of the map (width and height)
# - the function which represents the height of the map
# - the weight of the height difference in the reward
# - the weight of the constant step cost in the reward
# - the constant step cost
# - the starting position of the agent
# - the goal position of the agent

# Import the necessary packages
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Define the height function
def sample_terrain_function(x: int, y: int, height: int, width: int, mountain_height: float) -> \
        float:
    """
    The height function is such that along the y axis it is a parabola with maximum at the
    center and zeros at the beginning and end of the domain. This parabola is reduced by a
    linear function along the x axis. This is to represent a mountain range which is higher
    close to left edge of the map and lower close to the right edge of the map.
    :param x: the x coordinate
    :param y: the y coordinate
    :param height: the height of the map
    :param width: the width of the map
    :param mountain_height: the maximum height of the mountain

    :return: the height of the map at the given coordinates

    """
    # Define the y of the highest point of the mountain
    center_x = height / 2
    scaled_altitude_x = (1. - (x / center_x - 1.) ** 2)
    scaled_altitude = scaled_altitude_x * (1. - y / width)
    return mountain_height * scaled_altitude


# Define the class UnevenMaze
class UnevenMaze(gym.Env):
    """
    Description:
        A maze with an uneven surface. The agent is rewarded for reaching the goal and penalized for taking steps. The cost of the steps is a weighted sum of a constant step cost and the height difference between the start and end of the step. The weight is a parameter of the environment.
        Source:
    """

    def __init__(self, config: Dict):
        # Define the parameters of the environment
        self._config = config
        self.width: int = config['width']
        self.height: int = config['height']
        self.mountain_height: float = config['mountain_height']
        self._terrain_function: Callable = config['terrain_function']
        self.cost_height: float = config['cost_height']
        self.cost_height_max: float = config['cost_height_max']
        self.cost_step: float = config['cost_step']
        self.cost_step_max: float = config['cost_step_max']
        self._start_position: List[int, int] = config['start_position']
        self.goal_position: Tuple[int, int] = config['goal_position']
        self._max_steps: int = config['max_steps']
        self._current_step = 0
        self._current_position = copy.deepcopy(self._start_position)
        self._fig = None
        self._ax = None
        # Define the action space
        self.action_space = gym.spaces.Discrete(4)

        # Define the observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.cost_height_max, self.cost_step_max, self.width,
                           self.height]),
            dtype=np.float32
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        :param action: the action to take
        :return: the next observation, the reward, whether the episode is done, and the info
        """
        # Increment the step counter
        self._current_step += 1

        # Get the next position
        next_position = self._get_next_position(action)

        # Get the reward
        reward = self._get_reward(self.current_position, next_position)

        # Update the current position
        self._current_position = self._set_position(next_position)

        # Get the observation
        observation = self._get_observation()

        # Get the terminated and truncated flags
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Define the info
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        :return: the initial observation
        """
        # Reset the step counter
        self._current_step = 0

        # Reset the current position
        self._current_position = self._set_position(self._start_position)

        # Get the observation
        observation = self._get_observation()

        inforeset = {}

        return observation, inforeset

    def _get_next_position(self, action):
        """
        Get the next position.
        :param action: the action to take
        :return: the next position
        """
        # Get the next position
        next_position = self.current_position
        if action == 0:
            next_position[0] += 1
        elif action == 1:
            next_position[0] -= 1
        elif action == 2:
            next_position[1] += 1
        elif action == 3:
            next_position[1] -= 1
        else:
            raise ValueError('Invalid action.')

        # if the agent goes out of bounds of height or width, it stays in the same position
        if next_position[0] < 0 or next_position[0] >= self.height:
            next_position[0] = self.current_position[0]
        if next_position[1] < 0 or next_position[1] >= self.width:
            next_position[1] = self.current_position[1]

        return next_position

    def _get_reward(self, current_position: List[int], next_position: List[int]) -> \
            float:
        """
        Get the reward.
        :param current_position: the current position
        :param next_position: the next position
        :return: the reward
        """
        # Get the height of the current position
        current_height = self._get_altitude(current_position)

        # Get the height of the next position
        next_height = self._get_altitude(next_position)

        # Get the height difference
        height_difference = next_height - current_height

        # Only reward negatively for increasing height
        height_difference = height_difference if height_difference > 0 else 0.

        # Get the reward
        reward = - self.cost_height * height_difference - self.cost_step

        return reward

    def _set_position(self, position) -> np.ndarray:
        """
        Set the position.
        :param position: the position
        :return: the position
        """
        # Set the position
        self._current_position = copy.deepcopy(position)

        return self._current_position

    def _get_observation(self) -> np.ndarray:
        """
        Get the observation.
        :return: the observation
        """
        # Get the observation
        observation = np.array([
            self.cost_height,
            self.cost_step,
            self.current_position[0],
            self.current_position[1],
        ])

        return observation

    def render(self):
        """
        Rendering the environment as follows:
        - The agent is represented by a blue circle.
        - The goal is represented by a green circle.
        - The start is represented by a red circle.
        - The height of the terrain is represented by a color gradient of gray.

        :return:
        """
        # Define the color gradient
        color_gradient = np.linspace(0., 1., 256)
        color_gradient = np.vstack((color_gradient, color_gradient, color_gradient)).T

        # Define the figure
        if not self._fig:
            self._fig = plt.figure(figsize=(8, 8))
        if not self._ax:
            self._ax = self._fig.add_subplot(111)

        # Define the x and y coordinates
        altitudes = np.zeros(shape=(self.height, self.width))


        # Define the height
        for i in range(self.height):
            for j in range(self.width):
                altitudes[i, j] = self._get_altitude([i, j])

        # Plot the height
        self._ax.imshow(altitudes)

        # Plot the start
        self._ax.plot(self._start_position[1], self._start_position[0], 'ro', markersize=10)

        # Plot the goal
        self._ax.plot(self.goal_position[1], self.goal_position[0], 'go', markersize=10)

        # Plot the agent
        self._ax.plot(self.current_position[1], self.current_position[0], 'bo', markersize=10)

        # Set the title
        self._ax.set_title('Step: {}'.format(self._current_step))

        # Show the plot
        plt.show(block=False)
        plt.pause(0.1)

    def _get_altitude(self, position):
        """
        Get the altitude from the position
        :return: the height
        """
        altitude = self._terrain_function(position[0], position[1], self.height, self.width, self.mountain_height)
        return altitude

    def _get_terminated(self):
        """if the current position is the goal position, return True"""
        if self.current_position == self.goal_position:
            return True
        else:
            return False

    def _get_truncated(self):
        """if the current step is the maximum step, return True"""
        if self._current_step == self._max_steps:
            return True
        else:
            return False

    @property
    def config(self):
        return self._config

    @property
    def current_position(self):
        return copy.deepcopy(self._current_position)

