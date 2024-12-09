import time
import os
import random
from collections import deque
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecNormalize,
    DummyVecEnv,
)
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
import ale_py

gym.register_envs(ale_py)


@dataclass
class Args:
    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True

    # Environment Settings
    env_id: str = "ALE/Breakout-v5"
    n_envs: int = 1
    frame_stack: int = 4

    # Exploration Settings
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.1

    # Training Settings
    total_timesteps: int = 5000000
    buffer_size: int = 50000
    batch_size: int = 32
    learning_rate: float = 1e-4
    clip_grad_norm: int = 10
    weight_decay: int = 0
    gradient_steps: int = 2

    gamma: float = 0.99
    tau: float = 0.1
    train_frequency: int = 4
    target_update_frequency: int = 1000
    learning_start: int = 100

    # Saving & Logging Settings
    capture_video: bool = True
    video_dir: str = "videos"
    save_model: bool = True
    model_dir: str = "saved_models"
    model_save_frequency: int = 50000


args = Args()


def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def create_envs(
    env_id,
    n_envs,
    seed,
    frame_stack=4,
    normalize=False,
    normalize_kwargs=None,
):
    def make_env():
        env = gym.make(
            env_id,
            render_mode="rgb_array",
        )
        return env

    env = make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
        wrapper_class=AtariWrapper,
    )

    if normalize:
        normalize_kwargs = normalize_kwargs or {}
        env = VecNormalize(env, **normalize_kwargs)

    if frame_stack > 1:
        env = VecFrameStack(env, frame_stack)

    return env


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs

        self.observations = torch.zeros(
            (self.buffer_size, self.n_envs) + self.observation_space,
            dtype=torch.float32,
            device=device,
        )
        print(self.observations.shape)
        self.next_observations = torch.zeros(
            (self.buffer_size, self.n_envs) + self.observation_space,
            dtype=torch.float32,
            device=device,
        )
        self.actions = torch.zeros(
            (self.buffer_size, self.n_envs, self.action_space),
            dtype=torch.int64,
            device=device,
        )
        self.rewards = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.float32,
            device=device,
        )
        self.dones = torch.zeros(
            (self.buffer_size, self.n_envs),
            dtype=torch.bool,
            device=device,
        )
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done, info):
        self.observations[self.pos] = obs.detach().clone()
        self.next_observations[self.pos] = next_obs.detach().clone()
        self.actions[self.pos] = action.unsqueeze(-1).detach().clone()
        self.rewards[self.pos] = reward.detach().clone()
        self.dones[self.pos] = done.detach().clone()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        if self.full:
            batch_inds = (
                torch.randint(1, self.buffer_size, size=(batch_size,)) + self.pos
            ) % self.buffer_size
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))

        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        env_indices = torch.randint(0, high=self.n_envs, size=(len(batch_inds),))
        obs = self.observations[batch_inds, env_indices, :]
        print(obs.shape)
        next_obs = self.next_observations[batch_inds, env_indices, :]
        actions = self.actions[batch_inds, env_indices, :]
        rewards = self.rewards[batch_inds, env_indices].view(-1, 1)
        dones = self.dones[batch_inds, env_indices].int()
        return obs, actions, rewards, next_obs, dones


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        h, w, c = observation_space.shape
        action_size = action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(c, h, w)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, action_size)
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


def linear_schedule(start_e, end_e, duration, t):
    return max(end_e, start_e - (start_e - end_e) * (t / duration))


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    set_experiment_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    env = create_envs(
        args.env_id,
        n_envs=args.n_envs,
        seed=args.seed,
        frame_stack=args.frame_stack,
        normalize=True,
        normalize_kwargs={"gamma": 0.99, "clip_obs": 10.0},
    )

    q_network = QNetwork(env.observation_space, env.action_space).to(device)
    target_network = QNetwork(env.observation_space, env.action_space).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.AdamW(
        q_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        env.observation_space.shape,
        1,
        device=device,
        n_envs=args.n_envs,
    )

    pbar = tqdm(
        range(args.total_timesteps),
        total=args.total_timesteps,
        position=0,
    )
    scores = deque(maxlen=100)
    episode_scores = np.zeros(args.n_envs)
    obs = env.reset()
    obs = torch.from_numpy(obs).to(device)
    for global_step in pbar:
        epsilon = linear_schedule(
            args.start_epsilon,
            args.end_epsilon,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if np.random.random() < epsilon:
            action = np.array([env.action_space.sample() for _ in range(args.n_envs)])
        else:
            q_network.eval()
            with torch.no_grad():
                q_values = q_network(obs.permute(0, 3, 1, 2))
                action = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, reward, done, info = env.step(action)
        next_obs = torch.from_numpy(next_obs).to(device)
        action = torch.from_numpy(action).to(device)
        reward = torch.from_numpy(reward).to(device)
        done = torch.from_numpy(done).to(device)
        replay_buffer.add(obs, next_obs, action, reward, done, info)

        for i in range(args.n_envs):
            episode_scores[i] += reward[i].item()

            if done[i].item():
                scores.append(episode_scores[i])
                episode_scores[i] = 0

        obs = next_obs
        avg_score = np.mean(scores) if len(scores) > 0 else 0

        if (
            global_step > args.learning_start
            and global_step % args.train_frequency == 0
        ):
            losses = []
            q_network.train()
            for _ in range(args.gradient_steps):
                observations, actions, rewards, next_observations, dones = (
                    replay_buffer.sample(args.batch_size)
                )
                observations = observations.permute(0, 3, 1, 2)
                next_observations = next_observations.permute(0, 3, 1, 2)
                with torch.no_grad():
                    next_q_values = target_network(next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values = (
                        rewards.flatten()
                        * (1 - dones.flatten())
                        * args.gamma
                        * next_q_values.flatten()
                    ).unsqueeze(-1)

                current_q_values = q_network(observations)
                current_q_values = torch.gather(
                    current_q_values, dim=1, index=actions.long()
                )
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.cpu().item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    q_network.parameters(), max_norm=args.clip_grad_norm
                )
                optimizer.step()

            pbar.set_description(
                f"Avg Score: {avg_score:.2f}, Avg Loss: {np.round(np.mean(losses), 6) if len(losses) > 0 else 0}"
            )

            if global_step % args.target_update_frequency == 0:
                soft_update(q_network, target_network, args.tau)

            if args.save_model and global_step % args.model_save_frequency == 0:
                model_path = Path(args.model_dir) / f"dqn_{global_step}.pth"
                torch.save(q_network.state_dict(), model_path)
env.close()
