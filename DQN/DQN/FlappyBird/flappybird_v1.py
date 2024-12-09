import flappy_bird_gymnasium

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
from pathlib import Path


@dataclass
class Args:
    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 42
    cuda: bool = True

    # Environment Settings
    env_id: str = "FlappyBird-v0"
    num_envs: int = 1

    # Exploration Settings
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.1

    # Training Settings
    total_timesteps: int = 10000000
    buffer_size: int = 100000
    batch_size: int = 64
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.1
    train_frequency: int = 4
    target_update_frequency: int = 1000
    learning_start: int = 100000

    # Saving & Logging Settings
    capture_video: bool = True
    video_dir: str = "videos"
    save_model: bool = True
    model_dir: str = "saved_models"
    model_save_frequency: int = 50000


args = Args()


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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            normalize_obs=True,
            use_lidar=False,
            audio_on=True,
        )
        if capture_video and idx == 0:
            env = RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: (x + 1) % 500 == 0,
            )
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env: gym.Env, linear1_units=512, linear2_units=256):
        super().__init__()
        state_size = np.array(env.single_observation_space.shape).prod()
        action_size = env.single_action_space.n
        self.linear1 = nn.Linear(state_size, linear1_units)
        self.linear2 = nn.Linear(linear1_units, linear2_units)
        self.linear3 = nn.Linear(linear2_units, action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        # Initialize buffers for states, actions, rewards, next states, and dones
        self.states = torch.zeros(
            (buffer_size, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (buffer_size, *action_shape), dtype=torch.long, device=device
        )
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(
            (buffer_size, *obs_shape), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)

        self.idx = 0  # Current index in the buffer
        self.full = False  # Indicates whether the buffer is full

    def add(self, state, action, reward, next_state, done):
        # Add new experience to the buffer
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done

        # Update index and check if buffer is full
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True

    def sample(self):
        # Randomly sample indices for batch
        max_idx = self.buffer_size if self.full else self.idx
        indices = torch.randint(0, max_idx, (self.batch_size,), device=self.device)

        # Gather batch data
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        )

    def __len__(self):
        return self.buffer_size if self.full else self.idx


# Epsilon decay schedule
def linear_schedule(start_e, end_e, duration, t):
    return max(end_e, start_e - (start_e - end_e) * (t / duration))


# Soft update function
def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
set_experiment_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# Create directories
os.makedirs(args.video_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)


envs = gym.vector.SyncVectorEnv(
    [
        make_env(
            args.env_id,
            args.seed + i,
            i,
            args.capture_video,
            run_name,
        )
        for i in range(args.num_envs)
    ]
)

q_network = QNetwork(envs).to(device)
target_network = QNetwork(envs).to(device)
optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate)
target_network.load_state_dict(q_network.state_dict())

obs_shape = envs.single_observation_space.shape
action_shape = (1,)  # Assuming discrete action space
replay_buffer = ReplayBuffer(
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    obs_shape=obs_shape,
    action_shape=action_shape,
    device=device,  # Pass the GPU/CPU device
)

# Training loop
pbar = tqdm(range(args.total_timesteps), total=args.total_timesteps)
scores = deque(maxlen=100)

states, _ = envs.reset(seed=args.seed)
states = torch.tensor(states, dtype=torch.float32, device=device)
episode_scores = np.zeros(args.num_envs)

for global_step in pbar:
    epsilon = linear_schedule(
        args.start_epsilon,
        args.end_epsilon,
        args.exploration_fraction * args.total_timesteps,
        global_step,
    )

    # Select action
    if np.random.random() < epsilon:
        actions = np.array(
            [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        )
    else:
        with torch.no_grad():
            q_values = q_network(states)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

    # Step the environment
    next_states, rewards, terminations, truncations, infos = envs.step(actions)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(terminations | truncations, dtype=torch.float32, device=device)
    # Add experiences to the buffer
    for i in range(args.num_envs):
        replay_buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        episode_scores[i] += rewards[i].item()

        if dones[i].item():
            scores.append(episode_scores[i])
            episode_scores[i] = 0

    states = next_states
    avg_score = np.mean(scores) if len(scores) > 0 else 0
    pbar.set_description(f"Avg Score: {avg_score:.2f}")

    if global_step > args.learning_start and global_step % args.train_frequency == 0:
        states, actions, rewards, next_states, dones = replay_buffer.sample()
        mask = 1 - dones
        with torch.no_grad():
            max_next_q = target_network(next_states).max(1)[0]
            targets = rewards + args.gamma * max_next_q * mask

        q_values = q_network(states)  # Shape: (batch_size, num_actions)
        actions = actions.view(-1, 1)  # Ensure actions have shape (batch_size, 1)
        current_q = q_values.gather(1, actions).squeeze(-1)  # Shape: (batch_size,)

        loss = F.smooth_l1_loss(current_q, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10)
        optimizer.step()

    if global_step % args.target_update_frequency == 0:
        soft_update(q_network, target_network, args.tau)

    # Save the model periodically
    if args.save_model and global_step % args.model_save_frequency == 0:
        model_path = Path(args.model_dir) / f"dqn_{global_step}.pth"
        torch.save(q_network.state_dict(), model_path)

# Close the environment
envs.close()
