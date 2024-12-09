from collections import defaultdict
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Algorithm: Expected SARSA
# Environment Description: https://gymnasium.farama.org/environments/toy_text/cliff_walking/


class CliffWalkingAgent:
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

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def expected_q(self, next_obs):
        # Expected Q value calculation based on ε-greedy policy
        q_values_next = self.q_values[next_obs]
        best_action = np.argmax(q_values_next)
        num_actions = len(q_values_next)

        # Probability of choosing the best action (1 - ε) + ε / n
        policy_probabilities = np.ones(num_actions) * (self.epsilon / num_actions)
        policy_probabilities[best_action] += 1 - self.epsilon

        # Expected Q as weighted sum of Q-values
        expected_q_value = np.dot(policy_probabilities, q_values_next)
        return expected_q_value

    def update(self, obs, action, reward, terminated, next_obs):
        expected_q_value = self.expected_q(next_obs) if not terminated else 0
        temporal_difference = (
            reward
            + self.discount_factor * expected_q_value
            - self.q_values[obs][action]
        )
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# HYPERPARAMETERS
learning_rate = 0.01
n_episodes = 10000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1
discount_factor = 0.99

env = gym.make("CliffWalking-v0", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="cliffwalking-agent",
    episode_trigger=lambda x: (x + 1) % 100 == 0,
)
env = RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = CliffWalkingAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)

progress_bar = tqdm(range(n_episodes), total=n_episodes, position=0)
current_episode = 0
for episode in progress_bar:
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs=obs)
        next_obs, reward, terminated, truncated, info = env.step(action=action)
        agent.update(obs, action, reward, terminated, next_obs)
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()

env.close()

# print(f"Episode time taken: {env.time_queue}")
# print(f"Episode total rewards: {env.return_queue}")
# print(f"Episode lengths: {env.length_queue}")


# Create a figure with 1 row and 3 columns for side-by-side subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

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

# Save the figure
plt.savefig("agent_performance_metrics.png", dpi=300, bbox_inches="tight")
# plt.show()
