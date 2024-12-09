import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True
    save_model: bool = True

    env_id: str = "LunarLander-v3"
    total_timesteps = 5000
    learning_rate: float = 5e-4
    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 1e-3
    target_network_frequency: int = 1
    batch_size: int = 64
    start_epsilon: int = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.5
    learning_start: int = 100
    train_frequency: int = 4


def set_experiment_seed(seed=42):
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If using GPU, set random seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Set to False for reproducibility

    # Optionally log seed information
    print(f"Random seed set to {seed}")


def make_env(env_id, seed, capture_video, run_name):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            # env.observation_space.seed(seed)
            env = RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: (x + 1) % 1000 == 0,
            )
        else:
            env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env: gym.Env, linear1_units=120, linear2_units=84):
        super().__init__()
        state_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n
        self.linear1 = nn.Linear(state_size, linear1_units)
        self.linear2 = nn.Linear(linear1_units, linear2_units)
        self.linear3 = nn.Linear(linear2_units, action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)


def linear_schedule(
    start_e: float, end_e: float, duration: int, t: int, eps_decay: float = 0.995
):
    return max(end_e, start_e - (start_e - end_e) * (t / duration))


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


args = Args()
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
set_experiment_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
env = make_env(args.env_id, args.seed, args.capture_video, run_name)()

q_network = QNetwork(env).to(device)
target_network = QNetwork(env).to(device)
optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate)
target_network.load_state_dict(q_network.state_dict())

replay_buffer = ReplayBuffer(
    env.action_space.n, args.buffer_size, args.batch_size, args.seed, device
)

start_time = time.time()
scores = []  # list containing scores from each episode
scores_window = deque(maxlen=100)
pbar = tqdm(range(args.total_timesteps), total=args.total_timesteps, position=0)

for global_step in pbar:
    score = 0
    state, _ = env.reset(seed=args.seed)
    epsilon = linear_schedule(
        args.start_epsilon,
        args.end_epsilon,
        args.exploration_fraction * args.total_timesteps,
        global_step,
    )

    while True:  # loop for each step within the episode
        if np.random.random() < epsilon:
            action = random.choice(np.arange(env.action_space.n))
        else:
            q_values = q_network(
                torch.from_numpy(state).float().unsqueeze(0).to(device)
            )
            action = np.argmax(q_values.cpu().data.numpy())

        # Save data to replay buffer, handle termination
        next_state, reward, termination, truncation, info = env.step(action)
        score += reward  # accumulate reward per episode
        real_next_state = next_state.copy()
        if "final_observation" in info and truncation:
            real_next_state = info["final_observation"]

        done = termination or truncation
        replay_buffer.add(state, action, reward, real_next_state, done)

        # Set current state to next state
        state = next_state

        if done:
            scores.append(score)
            scores_window.append(score)
            avg_score = np.mean(scores_window)
            pbar.set_description(f"Avg Score: {avg_score:.2f}")
            break  # end of episode

        if global_step > args.learning_start:
            if global_step % args.train_frequency == 0:
                data = replay_buffer.sample()
                states, actions, rewards, next_states, dones = data

                Q_targets_next = (
                    target_network(next_states).detach().max(dim=1)[0].unsqueeze(1)
                )
                Q_targets = rewards + (args.gamma * Q_targets_next * (1 - dones))

                Q_expected = q_network(states).gather(1, actions)
                loss = F.mse_loss(Q_expected, Q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target network
                if global_step % args.target_network_frequency == 0:
                    soft_update(q_network, target_network, args.tau)

    if np.mean(scores_window) >= 200.0:
        print(
            "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                global_step - 100, np.mean(scores_window)
            )
        )
        break

# Save model if specified
if args.save_model:
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "q_network.pth")
    torch.save(q_network.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Close the environment
env.close()
