from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def discretize_state(state, bins):
    return tuple(np.digitize(s, bins[i]) for i, s in enumerate(state))


class CartPoleAgent:
    def __init__(
        self,
        env,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        self.epsilon_history = []  # To log epsilon decay

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self, episode):
        # Adaptive epsilon decay based on progress
        if episode < 0.5 * n_episodes:
            decay_rate = self.epsilon_decay
        else:
            decay_rate = self.epsilon_decay * 0.5
        self.epsilon = max(self.final_epsilon, self.epsilon * decay_rate)
        self.epsilon_history.append(self.epsilon)


# Increased discretization, especially in the pole angle and velocity dimensions
n_bins = [6, 12, 12, 20]  # More bins for pole angle and pole velocity

bins = [
    np.linspace(-4.8, 4.8, n_bins[0] - 1),  # Cart position
    np.linspace(-4, 4, n_bins[1] - 1),  # Cart velocity
    np.linspace(-0.418, 0.418, n_bins[2] - 1),  # Pole angle (more bins)
    np.linspace(-4, 4, n_bins[3] - 1),  # Pole velocity (more bins)
]

# HYPERPARAMETERS
learning_rate = 0.01
n_episodes = 50000
start_epsilon = 1.0
epsilon_decay = 0.999
final_epsilon = 0.1
discount_factor = 0.99

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = RecordEpisodeStatistics(env, buffer_length=n_episodes)
env = RecordVideo(
    env, video_folder="cartpole-agent", episode_trigger=lambda x: x % 5000 == 0
)

agent = CartPoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)

progress_bar = tqdm(range(n_episodes), total=n_episodes, position=0)
for episode in progress_bar:
    obs, info = env.reset()
    obs = discretize_state(obs, bins=bins)
    done = False
    while not done:
        action = agent.get_action(obs=obs)
        next_obs, reward, terminated, truncated, info = env.step(action=action)
        next_obs = discretize_state(next_obs, bins=bins)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon(episode)

env.close()

# Create a figure with 1 row and 3 columns for side-by-side subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 6), sharey=False)

axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout(pad=2.0)

# Epsilon Decay Plot
axs[3].plot(agent.epsilon_history)
axs[3].set_title("Epsilon Decay")
axs[3].set_xlabel("Episode")
axs[3].set_ylabel("Epsilon Value")

plt.tight_layout(pad=2.0)
plt.savefig("agent_performance_metrics.png", dpi=300, bbox_inches="tight")
# plt.show()
