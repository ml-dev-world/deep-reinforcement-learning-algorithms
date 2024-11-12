import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


class TileCoder:
    def __init__(self, position_limits, velocity_limits, num_tilings, bins_per_tiling):
        self.position_limits = position_limits
        self.velocity_limits = velocity_limits
        self.num_tilings = num_tilings
        self.bins_per_tiling = bins_per_tiling

        # Calculate offset for each tiling
        self.offsets = [
            (
                i
                / self.num_tilings
                * (position_limits[1] - position_limits[0])
                / bins_per_tiling[0],
                i
                / self.num_tilings
                * (velocity_limits[1] - velocity_limits[0])
                / bins_per_tiling[1],
            )
            for i in range(num_tilings)
        ]

    def get_tiles(self, position, velocity):
        features = []
        for tiling, offset in enumerate(self.offsets):
            position_bin = min(
                self.bins_per_tiling[0] - 1,
                max(
                    0,
                    int(
                        (position - self.position_limits[0] + offset[0])
                        / (self.position_limits[1] - self.position_limits[0])
                        * self.bins_per_tiling[0]
                    ),
                ),
            )
            velocity_bin = min(
                self.bins_per_tiling[1] - 1,
                max(
                    0,
                    int(
                        (velocity - self.velocity_limits[0] + offset[1])
                        / (self.velocity_limits[1] - self.velocity_limits[0])
                        * self.bins_per_tiling[1]
                    ),
                ),
            )

            feature_index = (
                tiling * (self.bins_per_tiling[0] * self.bins_per_tiling[1])
                + (position_bin * self.bins_per_tiling[1])
                + velocity_bin
            )
            features.append(feature_index)
        return features


class MountainCarAgent:
    def __init__(self, env, tile_coder, learning_rate, epsilon, discount_factor):
        self.env = env
        self.tile_coder = tile_coder
        self.q_weights = np.zeros(
            (
                env.action_space.n,
                tile_coder.num_tilings * np.prod(tile_coder.bins_per_tiling),
            )
        )
        self.lr = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor

    def get_action(self, position, velocity):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [
                self.get_q_value(position, velocity, a)
                for a in range(self.env.action_space.n)
            ]
            return int(np.argmax(q_values))

    def get_q_value(self, position, velocity, action):
        features = self.tile_coder.get_tiles(position, velocity)
        return np.sum(self.q_weights[action][features])

    def update(
        self,
        position,
        velocity,
        action,
        reward,
        next_position,
        next_velocity,
        next_action,
        terminated,
    ):
        current_q = self.get_q_value(position, velocity, action)
        next_q = (
            0
            if terminated
            else self.get_q_value(next_position, next_velocity, next_action)
        )
        td_error = reward + self.discount_factor * next_q - current_q
        features = self.tile_coder.get_tiles(position, velocity)
        for feature in features:
            self.q_weights[action][feature] += self.lr * td_error


# Parameters for Tile Coding and SARSA
num_tilings = 8
bins_per_tiling = [10, 10]
learning_rate = 0.1 / num_tilings
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.05
discount_factor = 0.99
n_episodes = 5000

# Initialize environment and agent
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
env = RecordVideo(
    env, video_folder="mountaincar-agent", episode_trigger=lambda x: (x + 1) % 1000 == 0
)

tile_coder = TileCoder(
    position_limits=(-1.2, 0.6),
    velocity_limits=(-0.07, 0.07),
    num_tilings=num_tilings,
    bins_per_tiling=bins_per_tiling,
)
agent = MountainCarAgent(env, tile_coder, learning_rate, epsilon, discount_factor)

# Training Loop
episode_rewards = []
for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    position, velocity = obs
    action = agent.get_action(position, velocity)
    done = False
    total_reward = 0

    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_position, next_velocity = next_obs
        next_action = agent.get_action(next_position, next_velocity)

        # SARSA update
        agent.update(
            position,
            velocity,
            action,
            reward,
            next_position,
            next_velocity,
            next_action,
            terminated,
        )

        position, velocity = next_position, next_velocity
        action = next_action
        total_reward += reward
        done = terminated or truncated

    # Epsilon decay
    agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

# Plotting the Rewards
plt.plot(np.convolve(episode_rewards, np.ones(100) / 100, mode="valid"))
plt.title("Episode Rewards over Time (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
