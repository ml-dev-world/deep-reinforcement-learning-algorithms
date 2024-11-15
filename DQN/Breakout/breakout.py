import os
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import logging
from pathlib import Path
import ale_py
import warnings
from gymnasium.vector import AsyncVectorEnv


warnings.filterwarnings("ignore")
gym.register_envs(ale_py)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU at index 0


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 42
    env_id: str = "ALE/Breakout-v5"
    num_envs: int = 2
    total_timesteps: int = 1000000
    buffer_size: int = 100000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.1
    target_update_frequency: int = 1000
    train_frequency: int = 4
    learning_start: int = 100
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.1
    capture_video: bool = True
    save_model: bool = True
    cuda: bool = True
    frame_stack: int = 4
    screen_size: int = 64
    video_dir: str = "videos"
    model_dir: str = "saved_models"
    model_save_frequency: int = 50000


args = Args()


# Set seeds for reproducibility
def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# Create the environment with preprocessing
def make_env(env_id, seed, idx, capture_video, screen_size, frame_stack, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)
        if capture_video and idx == 0:
            env = RecordVideo(
                env,
                os.path.join(args.video_dir, run_name),
                episode_trigger=lambda x: x % 100 == 0,
            )
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (screen_size, screen_size))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, frame_stack)
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        c, h, w = env.single_observation_space.shape
        action_size = env.single_action_space.n

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically compute the size of the flattened output after convolutions
        conv_out_size = self._get_conv_out(c, h, w)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, action_size)
        )

    def _get_conv_out(self, c, h, w):
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            output = self.features(dummy_input)
            return output.numel()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class EfficientReplayBuffer:
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
            args.screen_size,
            args.frame_stack,
            run_name,
        )
        for i in range(args.num_envs)
    ]
)

q_network = QNetwork(envs).to(device)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

# Initialize the replay buffer
obs_shape = envs.single_observation_space.shape
action_shape = (1,)  # Assuming discrete action space
replay_buffer = EfficientReplayBuffer(
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
states = torch.tensor(states, dtype=torch.float32, device=device).div(255.0)
episode_scores = np.zeros(args.num_envs)  # Track scores for each environment

for global_step in pbar:
    epsilon = linear_schedule(
        args.start_epsilon,
        args.end_epsilon,
        args.exploration_fraction * args.total_timesteps,
        global_step,
    )

    # Select action
    if random.random() < epsilon:
        actions = np.array(
            [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        )
    else:
        with torch.no_grad():
            q_values = q_network(states)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

    # Step the environment
    next_states, rewards, terminations, truncations, infos = envs.step(actions)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device).div(
        255.0
    )
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
        with torch.no_grad():
            max_next_q = target_network(next_states).max(1)[0]
            targets = rewards + args.gamma * (1 - dones) * max_next_q

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
