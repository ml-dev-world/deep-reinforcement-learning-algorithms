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
    end_epsilon: float = 0.05
    exploration_fraction: float = 0.1

    # Training Settings
    total_timesteps: int = 100000
    buffer_size: int = 100000
    batch_size: int = 32
    learning_rate: float = 0.00023779804030068177
    gamma: float = 0.9064052558039715
    tau: float = 0.3501545989692456
    train_frequency: int = 2
    target_update_frequency: int = 10
    learning_start: int = 500

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

    # If using GPU, set random seed for
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Optionally log seed information
    print(f"Random seed set to {seed}")


def make_env(env_id, seed, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id,
            render_mode="rgb_array",
            use_lidar=False,
            audio_on=True,
        )
        if capture_video:
            env = RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: (x + 1) % 1000 == 0,
            )
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class DQN(nn.Module):
    def __init__(self, env: gym.Env, linear1_units=128, linear2_units=512):
        super().__init__()
        state_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n

        self.linear1 = nn.utils.parametrizations.weight_norm(
            nn.Linear(state_size, linear1_units)
        )
        self.ln1 = nn.LayerNorm(linear1_units)

        self.linear2 = nn.utils.parametrizations.weight_norm(
            nn.Linear(linear1_units, linear2_units)
        )
        self.ln2 = nn.LayerNorm(linear2_units)

        self.output_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(linear2_units, action_size)
        )

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = F.leaky_relu(self.ln1(self.linear1(state)))
        x = self.dropout(x)
        x = F.leaky_relu(self.ln2(self.linear2(x)))
        return self.output_layer(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.states = torch.zeros(
            (buffer_size, *obs_shape), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(
            (buffer_size, *obs_shape), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)

        self.idx = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        # Add new experience to the buffer
        self.states[self.idx] = torch.tensor(
            state, device=self.device, dtype=torch.float32
        )
        self.actions[self.idx] = torch.tensor(
            action, device=self.device, dtype=torch.long
        )
        self.rewards[self.idx] = torch.tensor(
            reward, device=self.device, dtype=torch.float32
        )
        self.next_states[self.idx] = torch.tensor(
            next_state, device=self.device, dtype=torch.float32
        )
        self.dones[self.idx] = torch.tensor(
            done, device=self.device, dtype=torch.float32
        )

        # Update index and check if buffer is full
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True

    def sample(self):
        max_idx = self.buffer_size if self.full else self.idx
        indices = torch.randint(0, max_idx, (self.batch_size,), device=self.device)

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


def linear_schedule(start_e, end_e, duration, t):
    return max(end_e, start_e - (t / duration) * (start_e - end_e))


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

env = make_env(
    args.env_id,
    args.seed,
    args.capture_video,
    run_name,
)()


q_network = DQN(env).to(device)
target_network = DQN(env).to(device)
optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate)
target_network.load_state_dict(q_network.state_dict())

obs_shape = env.observation_space.shape
action_shape = (1,)  # Assuming discrete action space
replay_buffer = ReplayBuffer(
    buffer_size=args.buffer_size,
    batch_size=args.batch_size,
    obs_shape=obs_shape,
    action_shape=action_shape,
    device=device,  # Pass the GPU/CPU device
)


pbar = tqdm(range(args.total_timesteps), total=args.total_timesteps)
scores = []
scores_window = deque(maxlen=100)
best_avg_score = -float("inf")
model_save_path = os.path.join(args.model_dir, "best_q_network.pth")

for global_step in pbar:
    score = 0
    state, _ = env.reset()
    epsilon = linear_schedule(
        args.start_epsilon,
        args.end_epsilon,
        args.total_timesteps,
        global_step,
    )
    while True:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(
                    torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                )
                action = torch.argmax(q_values).item()

        next_state, reward, termination, truncation, info = env.step(action)
        real_next_state = next_state.copy()
        if truncation and "final_observation" in info:
            real_next_state = info["final_observation"]

        score += reward
        done = termination or truncation
        replay_buffer.add(state, action, reward, real_next_state, done)

        state = next_state

        if done:
            scores.append(score)
            scores_window.append(score)
            avg_score = np.mean(scores_window)
            pbar.set_description(f"Avg Score: {avg_score:.3f}")

            save_threshold = 1.0
            if (
                avg_score > best_avg_score
                and (avg_score - best_avg_score) > save_threshold
                and global_step >= args.learning_start
            ):
                best_avg_score = avg_score
                torch.save(q_network.state_dict(), model_save_path)
                print(f"New best model saved with Avg Score: {avg_score:.2f}")
            break
        if len(replay_buffer) >= args.batch_size:
            if global_step % args.train_frequency == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample()

                with torch.no_grad():
                    q_targets_next = target_network(next_states).max(dim=1)[0]
                    q_targets = rewards + ((args.gamma * q_targets_next) * (1 - dones))

                q_expected = (
                    q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                )
                loss = F.smooth_l1_loss(q_expected, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % args.target_update_frequency == 0:
                    soft_update(q_network, target_network, args.tau)

env.close()
