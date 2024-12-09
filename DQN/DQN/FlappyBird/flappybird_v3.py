import flappy_bird_gymnasium

import os
import random
import time
from dataclasses import dataclass
from collections import deque, namedtuple
from tqdm import tqdm

import optuna
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
    total_timesteps: int = 50000
    buffer_size: int = 100000
    batch_size: int = 32
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.1
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
            pass
            # env = RecordVideo(
            #     env,
            #     f"videos/{run_name}",
            #     episode_trigger=lambda x: (x + 1) % 1000 == 0,
            # )
        env = RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class DQN(nn.Module):
    def __init__(self, env: gym.Env, linear1_units=512, linear2_units=256):
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


def objective(trial):
    # Sample hyperparameters
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    args.gamma = trial.suggest_float("gamma", 0.9, 0.999)
    args.tau = trial.suggest_float("tau", 0.05, 0.5)
    args.batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    args.linear1_units = trial.suggest_int("linear1_units", 128, 1024, step=128)
    args.linear2_units = trial.suggest_int("linear2_units", 64, 512, step=64)

    # Reinitialize environment and networks
    env = make_env(args.env_id, args.seed, args.capture_video, run_name)()
    obs_shape = env.observation_space.shape

    q_network = DQN(env, args.linear1_units, args.linear2_units).to(device)
    target_network = DQN(env, args.linear1_units, args.linear2_units).to(device)
    optimizer = optim.AdamW(q_network.parameters(), lr=args.learning_rate)
    target_network.load_state_dict(q_network.state_dict())
    replay_buffer = ReplayBuffer(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        obs_shape=obs_shape,
        action_shape=(1,),
        device=device,
    )

    # Training loop
    scores_window = deque(maxlen=100)
    for global_step in tqdm(range(args.total_timesteps), total=args.total_timesteps):
        score = 0
        state, _ = env.reset()
        epsilon = linear_schedule(
            args.start_epsilon, args.end_epsilon, args.total_timesteps, global_step
        )

        while True:
            # Select action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(
                        torch.tensor(
                            state, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                    )
                    action = torch.argmax(q_values).item()

            # Take action and store experience
            next_state, reward, termination, truncation, info = env.step(action)
            real_next_state = next_state.copy()
            if truncation and "final_observation" in info:
                real_next_state = info["final_observation"]

            score += reward
            done = termination or truncation
            replay_buffer.add(state, action, reward, real_next_state, done)
            state = next_state

            if done:
                break

            # Training
            if (
                len(replay_buffer) >= args.batch_size
                and global_step % args.train_frequency == 0
            ):
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

        # Track scores
        scores_window.append(score)
        avg_score = np.mean(scores_window)

        # Report to Optuna and prune if necessary
        trial.report(avg_score, global_step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    env.close()
    return np.mean(scores_window)


# Optimize with Optuna
study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.HyperbandPruner()
)

study.optimize(objective, n_trials=50, timeout=3600)  # 50 trials or 1-hour limit

# Print best hyperparameters
print("Best Hyperparameters:")
print(study.best_params)
