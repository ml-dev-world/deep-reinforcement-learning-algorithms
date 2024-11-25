import os
import random
import time
import logging
from colorama import Fore, Style
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from tabulate import tabulate

import gymnasium as gym
import ale_py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Register Atari environments from ALE
gym.register_envs(ale_py)


class ColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    RESET = Style.RESET_ALL  # Reset color to default

    def format(self, record):
        # Add color to log level
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"

        # Add color to message
        record.msg = f"{log_color}{record.msg}{self.RESET}"

        return super().format(record)


def setup_logger(name="root", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Set custom color formatter
    formatter = ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


logger = setup_logger()


class Timer:
    def __init__(self, name="Timer", logger=None):
        self.name = name
        self.logger = logger
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{self.name} started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.logger.info(
            f"{self.name} finished. Elapsed time: {elapsed_time:.2f} seconds."
        )


@dataclass
class Args:
    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 42
    cuda: bool = True

    # Environment Settings
    env_id: str = "ALE/Breakout-v5"
    num_envs: int = 1
    frame_stack: int = 4

    # Exploration Settings
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    exploration_fraction: float = 0.1
    decay_rate = 150000

    # Training Settings
    total_timesteps: int = 10000000
    buffer_size: int = 60000
    gamma: float = 0.99
    tau: float = 1.0
    train_frequency: int = 4
    target_update_frequency: int = 1000
    learning_start: int = 50000

    # Model settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    clip_grad_norm: int = 10
    weight_decay: int = 0
    gradient_steps: int = 2

    # Saving & Logging Settings
    capture_video: bool = True
    video_dir: str = "videos"
    save_model: bool = True
    model_dir: str = "saved_models"
    model_save_frequency: int = 50000
    log_interval = 10000


def log_args(args: Args):
    logger.info("Experiment Configuration:")
    for field in args.__dataclass_fields__:
        value = getattr(args, field)
        logger.info(f"{field}: {value}")


def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id, render_mode="rgb_array" if capture_video and idx == 0 else None
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=f"videos/{run_name}",
            )

        # Add wrappers
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=1,
            screen_size=(84, 84),
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True,
            terminal_on_life_loss=False,
        )
        env = gym.wrappers.FrameStackObservation(env, 4)

        # Set Seed
        env.action_space.seed(seed)
        return env

    return thunk


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.n_envs = n_envs
        self.ReplayBufferSample = namedtuple(
            "Observation", ["state", "actions", "rewards", "next_state", "dones"]
        )

        self.states = torch.zeros(
            (self.buffer_size,) + self.observation_space,
            dtype=torch.float32,
            device=device,
        )
        self.next_states = torch.zeros(
            (self.buffer_size,) + self.observation_space,
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

    def add(self, state, next_state, action, reward, done, info):
        self.states[self.pos] = state.detach().clone()
        self.next_states[self.pos] = next_state.detach().clone()
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
        state = self.states[batch_inds, env_indices, :]
        next_state = self.next_states[batch_inds, env_indices, :]
        actions = self.actions[batch_inds, env_indices, :]
        rewards = self.rewards[batch_inds, env_indices].view(-1, 1)
        dones = self.dones[batch_inds, env_indices].int()
        return self.ReplayBufferSample(
            state=state,
            actions=actions,
            rewards=rewards,
            next_state=next_state,
            dones=dones,
        )

    def size(self):
        return self.buffer_size if self.full else self.pos


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        c, h, w = observation_space.shape
        action_size = action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(c, h, w)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, action_size)
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def exponential_schedule(start_e: float, end_e: float, decay_rate: float, t: int):
    return end_e + (start_e - end_e) * np.exp(-t / decay_rate)


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def initialize_environment(
    env_id: str, seed: int, num_envs: int, capture_video: bool, run_name: str
):
    return gym.vector.SyncVectorEnv(
        [
            make_env(env_id, seed + i, i, capture_video, run_name)
            for i in range(num_envs)
        ]
    )


def initialize_q_networks(observation_space, action_space, device: torch.device):
    q_network = QNetwork(
        observation_space=observation_space, action_space=action_space
    ).to(device)
    target_network = QNetwork(
        observation_space=observation_space, action_space=action_space
    ).to(device)
    target_network.load_state_dict(
        q_network.state_dict()
    )  # Sync target network with Q-network
    return q_network, target_network


def initialize_replay_buffer(args, observation_space_shape, device: torch.device):
    return ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=observation_space_shape,
        action_space=1,  # Assuming discrete action space with 1 action dimension
        device=device,
        n_envs=args.num_envs,
    )


if __name__ == "__main__":
    args = Args()
    log_args(args)
    logger.info(
        f"Experiment Configuration: {args.exp_name} | Seed: {args.seed} | CUDA: {args.cuda}"
    )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    set_experiment_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device set to: {device.type}")

    # Environment setup
    logger.info(f"Initializing {args.num_envs} environments for {args.env_id}...")
    envs = initialize_environment(
        env_id=args.env_id,
        seed=args.seed,
        num_envs=args.num_envs,
        capture_video=args.capture_video,
        run_name=run_name,
    )
    logger.info(
        f"Environment creation completed: {args.env_id} | Environments: {args.num_envs}"
    )

    # Q-network and target network setup
    logger.info(f"Initializing Q-network and target network...")
    q_network, target_network = initialize_q_networks(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
    )
    logger.info(
        f"Q-network initialized with observation space: {envs.single_observation_space.shape} | "
        f"action space: {envs.single_action_space.n}"
    )

    # Optimizer setup
    logger.info(f"Setting up AdamW optimizer for Q-network...")
    optimizer = torch.optim.AdamW(
        q_network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    logger.info(
        f"Optimizer initialized with learning rate: {args.learning_rate} | Weight decay: {args.weight_decay}"
    )

    # Learning rate scheduler setup
    logger.info(f"Initializing learning rate scheduler...")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_timesteps, eta_min=1e-6
    )
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=25000,  # Adjust the LR every 50k steps
    #     gamma=0.9,  # Reduce LR by 50% each step
    # )
    logger.info(f"LR Scheduler initialized: StepLR | Step size: 50000 | Gamma: 0.5")

    # Replay buffer setup
    logger.info(f"Initializing replay buffer with buffer size: {args.buffer_size}...")
    replay_buffer = initialize_replay_buffer(
        args=args,
        observation_space_shape=envs.observation_space.shape,
        device=device,
    )
    logger.info("Replay Buffer initialized.")
    logger.info(
        f"Replay Buffer initialized with capacity: {args.buffer_size} | Observation space shape: {envs.observation_space.shape}"
    )

    network_updates = 0
    exploration_actions = 0
    exploitation_actions = 0
    episode_rewards = []
    losses = []
    episode_lengths = []
    cumulative_rewards = np.zeros(envs.num_envs)
    current_episode_lengths = np.zeros(envs.num_envs, dtype=int)

    with Timer("Training Loop Timer", logger):
        state, _ = envs.reset(seed=args.seed)
        state = torch.from_numpy(state).to(device)
        with tqdm(
            range(args.total_timesteps),
            total=args.total_timesteps,
            desc=f"{Fore.RED}Training Progress{Style.RESET_ALL}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for global_step in pbar:
                # Calculate epsilon
                epsilon = exponential_schedule(
                    start_e=args.start_epsilon,
                    end_e=args.end_epsilon,
                    decay_rate=args.decay_rate,
                    # duration=args.total_timesteps * args.exploration_fraction,
                    t=global_step,
                )

                # Exploration or Exploitation
                if np.random.random() < epsilon:
                    actions = np.array(
                        [
                            envs.single_action_space.sample()
                            for _ in range(envs.num_envs)
                        ]
                    )
                    exploration_actions += 1
                else:
                    q_values = q_network(state)
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
                    exploitation_actions += 1

                # Take a step in environment
                next_states, rewards, terminations, truncations, infos = envs.step(
                    actions
                )

                # Handle episode end
                dones = terminations | truncations
                real_next_states = next_states.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_states[idx] = infos["final_observation"][idx]

                # Update cumulative reward
                cumulative_rewards += rewards
                current_episode_lengths += 1
                for idx, done in enumerate(dones):
                    if done:
                        episode_rewards.append(cumulative_rewards[idx])
                        episode_lengths.append(current_episode_lengths[idx])
                        current_episode_lengths[idx] = 0
                        cumulative_rewards[idx] = 0

                # Prepare for replay buffer
                next_states = torch.from_numpy(real_next_states).to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards = torch.from_numpy(rewards).to(device)
                dones = torch.from_numpy(dones).to(device)
                replay_buffer.add(state, next_states, actions, rewards, dones, infos)

                state = next_states

                # Training the network
                if global_step > args.learning_start:
                    if global_step % args.train_frequency == 0:
                        for _ in range(args.gradient_steps):
                            # Sample a batch from the replay buffer
                            data = replay_buffer.sample(args.batch_size)

                            # No need to calculate gradients for the target network
                            with torch.no_grad():
                                target_max, _ = target_network(data.next_state).max(
                                    dim=1
                                )
                                target_qvalues = (
                                    data.rewards.flatten()
                                    + args.gamma
                                    * target_max
                                    * (1 - data.dones.flatten())
                                )

                            # Calculate current Q Values
                            current_qvalues = (
                                q_network(data.state).gather(1, data.actions).squeeze()
                            )

                            # Compute loss
                            loss = F.smooth_l1_loss(
                                target=target_qvalues, input=current_qvalues
                            )
                            losses.append(loss.item())

                            # Backpropagation
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                q_network.parameters(), max_norm=args.clip_grad_norm
                            )
                            optimizer.step()
                            lr_scheduler.step()

                            # Track network updates
                            network_updates += 1

                    if global_step % args.target_update_frequency == 0:
                        soft_update(
                            source=q_network, target=target_network, tau=args.tau
                        )

                    if global_step % args.log_interval == 0 and global_step > 0:
                        avg_loss = np.mean(losses) if losses else 0
                        current_lr = optimizer.param_groups[0]["lr"]
                        avg_reward = (
                            np.mean(episode_rewards[-20:]) if episode_rewards else 0
                        )
                        avg_episode_length = (
                            np.mean(episode_lengths[-20:]) if episode_lengths else 0
                        )

                        table_data = [
                            ["Step", global_step],
                            ["Epsilon", f"{epsilon:.4f}"],
                            ["Network Updates", network_updates],
                            ["Avg loss", f"{avg_loss:.6f}"],
                            ["Learning Rate", f"{current_lr:.7f}"],
                            ["Avg Reward (Last 20)", f"{avg_reward:.2f}"],
                            ["Exploration Actions", f"{exploration_actions}"],
                            ["Exploitation Actions", f"{exploitation_actions}"],
                            ["Avg. Episode Lengths (Last 20)", f"{avg_episode_length}"],
                            ["Replay Buffer Size", f"{replay_buffer.size()}"],
                        ]

                        print(
                            "\n"
                            + tabulate(
                                table_data, headers=["Metric", "Value"], tablefmt="grid"
                            )
                        )

                        losses = []

                    if args.save_model and global_step % args.model_save_frequency == 0:
                        model_path = Path(args.model_dir) / f"dqn_{global_step}.pth"
                        torch.save(q_network.state_dict(), model_path)
                        logger.debug(f"Saved {global_step} model to {args.model_dir}")
envs.close()
