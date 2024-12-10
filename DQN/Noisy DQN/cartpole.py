import os
import random
import time
import logging
from colorama import Fore, Style
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from tabulate import tabulate
from thop import profile
import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


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
    cuda: bool = True  # CartPole is simple and doesn't require a GPU

    # Environment Settings
    env_id: str = "CartPole-v1"

    # Exploration Settings
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01
    decay_rate: int = 200
    exploration_fraction: float = 0.75

    # Replay Buffer Settings
    buffer_size: int = 100000
    eps: float = 1e-2
    alpha: float = 0.6
    beta: float = 0.4

    # Training Settings
    total_timesteps: int = 10000
    gamma: float = 0.99
    tau: float = 1e-2
    train_frequency: int = 2
    target_update_frequency: int = 1
    learning_start: int = 100

    # Model Settings
    batch_size: int = 256
    learning_rate: float = 1e-4
    clip_grad_norm: float = 10.0
    weight_decay: float = 0.0
    gradient_steps: int = 1

    # Saving & Logging Settings
    capture_video: bool = False
    video_dir: str = "videos"
    save_model: bool = True
    model_dir: str = "models"
    model_save_frequency: int = 10000
    log_interval: int = 500


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


def make_env(env_id, seed, capture_video, run_name):
    # Create environment with appropriate render mode
    env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)

    # Add RecordVideo wrapper if capture_video is enabled
    if capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=f"videos/{run_name}",
            episode_trigger=lambda x: (x + 1) % 100 == 0,
        )

    # Add RecordEpisodeStatistics wrapper
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Set seed for reproducibility
    env.action_space.seed(seed)
    return env


class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1
        change = value - self.nodes[idx]
        self.nodes[idx] = value

        while idx >= 0:
            idx = (idx - 1) // 2
            self.nodes[idx] += change

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left = 2 * idx + 1
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                cumsum -= self.nodes[left]
                idx = 2 * idx + 2
        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        eps=1e-2,
        alpha=0.1,
        beta=0.5,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps

        self.tree = SumTree(size=self.buffer_size)

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
            (self.buffer_size, self.action_space),
            dtype=torch.long,
            device=device,
        )
        self.rewards = torch.zeros(
            (self.buffer_size,),
            dtype=torch.float32,
            device=device,
        )
        self.dones = torch.zeros(
            (self.buffer_size,),
            dtype=torch.bool,
            device=device,
        )
        self.pos = 0
        self.real_size = 0
        self.full = False

    def add(self, state, next_state, action, reward, done, info):
        self.tree.add(self.max_priority, self.pos)
        self.states[self.pos] = state.clone()
        self.next_states[self.pos] = next_state.clone()
        self.actions[self.pos] = action.unsqueeze(-1).clone()
        self.rewards[self.pos] = reward.clone()
        self.dones[self.pos] = done.clone()
        self.pos = (self.pos + 1) % self.buffer_size
        self.real_size = min(self.real_size + 1, self.buffer_size)
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        return self._get_samples(sample_idxs, weights, tree_idxs)

    def _get_samples(self, batch_inds, weights, tree_idxs):
        state = self.states[batch_inds]
        next_state = self.next_states[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds].view(-1, 1)
        dones = self.dones[batch_inds].int()
        return (
            self.ReplayBufferSample(
                state=state,
                actions=actions,
                rewards=rewards,
                next_state=next_state,
                dones=dones,
            ),
            weights,
            tree_idxs,
        )

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        return self.buffer_size if self.full else self.pos


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / self.in_features**0.5
        # Initialization for Mu
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        # Initialization for Sigma
        nn.init.constant_(self.weight_sigma, 0.5 / self.in_features**0.5)
        nn.init.constant_(self.bias_sigma, 0.5 / self.in_features**0.5)

    def reset_noise(self):
        # Generate random noise using factorized Gaussian noise
        def scale_noise(size):
            noise = torch.randn(size)
            return noise.sign() * torch.sqrt(noise.abs())

        self.weight_epsilon.copy_(scale_noise((self.out_features, self.in_features)))
        self.bias_epsilon.copy_(scale_noise(self.out_features))

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    def __init__(self, env: gym.Env, linear1_units=256, linear2_units=128):
        super().__init__()
        state_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n

        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, linear1_units),
            nn.LeakyReLU(),
        )

        # Advantage layer
        self.advantage_layer = nn.Sequential(
            NoisyLinear(linear1_units, linear2_units),
            nn.LeakyReLU(),
            NoisyLinear(linear2_units, action_size),  # action_size output
        )

        # Value layer
        self.value_layer = nn.Sequential(
            NoisyLinear(linear1_units, linear2_units),
            nn.LeakyReLU(),
            NoisyLinear(linear2_units, 1),  # Scalar output
        )

    def forward(self, x):
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        # Combine value and advantage to form Q
        q = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q

    def reset_noise(self):
        # Reset noise in all noisy layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ModelMetrics:
    def __init__(self, model, observation_space):
        self.model = model
        self.observation_sample = torch.randn((1,) + observation_space.shape)

    def calculate_flops(self, input_size, device="cpu"):
        self.model.to(device)
        input_tensor = torch.randn(*input_size).to(
            device
        )  # Generate random input tensor
        flops, _ = profile(self.model, inputs=(input_tensor,))
        return flops

    def calculate_disk_size(self):
        # Calculate model disk size in bytes
        disk_size_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        # Convert bytes to megabytes
        disk_size_mb = disk_size_bytes / (1024 * 1024)
        return disk_size_mb

    def get_model_metrics(self, device="cpu"):
        # Total parameters and trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Calculate FLOPs and disk size
        flops = self.calculate_flops(
            (1,) + self.observation_sample.shape, device=device
        )
        disk_size = self.calculate_disk_size()

        return {
            "num_params": num_params,
            "num_trainable_params": num_trainable_params,
            "flops": flops,
            "disk_size": disk_size,
        }


def exponential_schedule(start_e: float, end_e: float, decay_rate: float, t: int):
    return end_e + (start_e - end_e) * np.exp(-t / decay_rate)


def linear_schedule(
    start_e: float, end_e: float, duration: int, t: int, eps_decay: float = 0.995
):
    return max(end_e, start_e - (start_e - end_e) * (t / duration))


def polyak_averaging(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def initialize_environment(env_id: str, seed: int, capture_video: bool, run_name: str):
    return make_env(
        env_id=env_id,
        seed=seed,
        capture_video=capture_video,
        run_name=run_name,
    )


def initialize_q_networks(env: gym.Env, device: torch.device):
    q_network = NoisyQNetwork(env=env).to(device)
    target_network = NoisyQNetwork(env=env).to(device)
    target_network.eval()
    target_network.load_state_dict(
        q_network.state_dict()
    )  # Sync target network with Q-network
    return q_network, target_network


def initialize_replay_buffer(
    buffer_size,
    observation_space_shape,
    device,
    eps,
    alpha,
    beta,
):
    return PrioritizedReplayBuffer(
        buffer_size=buffer_size,
        observation_space=observation_space_shape,
        action_space=1,
        device=device,
        eps=eps,
        alpha=alpha,
        beta=beta,
    )


if __name__ == "__main__":
    args = Args()
    log_args(args)
    logger.info(
        f"Experiment Configuration: {args.exp_name} | Seed: {args.seed} | CUDA: {args.cuda}"
    )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    set_experiment_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    logger.info(f"Device set to: {device.type}")

    # Make dirs
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Environment setup
    logger.info(f"Initializing environment for {args.env_id}...")
    env = initialize_environment(
        env_id=args.env_id,
        seed=args.seed,
        capture_video=args.capture_video,
        run_name=run_name,
    )
    logger.info(f"Environment creation completed: {args.env_id}")

    # Q-network and target network setup
    logger.info(f"Initializing Q-network and target network...")
    q_network, target_network = initialize_q_networks(
        env=env,
        device=device,
    )
    logger.info(
        f"Q-network initialized with observation space: {env.observation_space.shape} | "
        f"action space: {env.action_space.n}"
    )
    model_metrics = ModelMetrics(
        model=q_network, observation_space=env.observation_space
    )
    metrics = model_metrics.get_model_metrics(device=device)
    logger.info(
        f"Num Params: {metrics['num_params']}, "
        f"Trainable Params: {metrics['num_trainable_params']}, "
        f"FLOPS: {metrics['flops']}, "
        f"Disk Size: {metrics['disk_size']:.3f} MB"
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
    logger.info(f"LR Scheduler initialized: StepLR | Step size: 50000 | Gamma: 0.5")

    # Prioritized Replay buffer setup
    logger.info(
        f"Initializing prioritized replay buffer with buffer size: {args.buffer_size}..."
    )
    replay_buffer = initialize_replay_buffer(
        buffer_size=args.buffer_size,
        observation_space_shape=env.observation_space.shape,
        device=device,
        eps=args.eps,
        alpha=args.alpha,
        beta=args.beta,
    )
    logger.info("Replay Buffer initialized.")
    logger.info(
        f"Replay Buffer initialized with capacity: {args.buffer_size} | Observation space shape: {env.observation_space.shape}"
    )
    logger.info(
        f"{replay_buffer.states.shape}, {replay_buffer.actions.shape},{replay_buffer.rewards.shape}"
    )

    network_updates = 0
    exploration_actions = 0
    exploitation_actions = 0
    target_network_updates = 0
    episode_rewards = []
    losses = []
    episode_lengths = []

    with Timer("Training Loop Timer", logger):
        with tqdm(
            range(args.total_timesteps),
            total=args.total_timesteps,
            desc=f"{Fore.RED}Training Progress{Style.RESET_ALL}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for global_step in pbar:
                # Initialize stats for the current episode
                current_episode_reward = 0
                current_episode_length = 0

                # Reset environment state
                state, _ = env.reset(seed=args.seed)
                state = torch.tensor(state, dtype=torch.float32, device=device)

                while True:
                    q_values = q_network(state.unsqueeze(0))
                    action = torch.argmax(q_values).cpu().numpy()

                    # Take a step in the environment
                    next_state, reward, termination, truncation, info = env.step(action)
                    done = termination or truncation
                    real_next_state = next_state.copy()

                    # Handle truncation for final observations
                    if "final_observation" in info:
                        real_next_state = info["final_observation"]

                    # Update episode statistics
                    current_episode_reward += reward
                    current_episode_length += 1

                    # Prepare and add transition to replay buffer
                    next_state_tensor = torch.tensor(
                        real_next_state, dtype=torch.float32, device=device
                    )
                    action_tensor = torch.tensor(
                        action, dtype=torch.long, device=device
                    )
                    reward_tensor = torch.tensor(
                        reward, dtype=torch.float32, device=device
                    )
                    done_tensor = torch.tensor(done, dtype=torch.bool, device=device)

                    replay_buffer.add(
                        state,
                        next_state_tensor,
                        action_tensor,
                        reward_tensor,
                        done_tensor,
                        info,
                    )

                    state = next_state_tensor

                    # Break loop if episode ends
                    if done:
                        episode_rewards.append(current_episode_reward)
                        episode_lengths.append(current_episode_length)
                        break

                # Train the network if conditions are met
                if (
                    replay_buffer.size() > args.batch_size
                    and global_step > args.learning_start
                    and global_step % args.train_frequency == 0
                ):
                    for _ in range(args.gradient_steps):
                        # Sample batch from replay buffer
                        data, weights, tree_idxs = replay_buffer.sample(args.batch_size)

                        with torch.no_grad():
                            # Double Q-Learning target calculation
                            next_actions = q_network(data.next_state).argmax(
                                dim=1, keepdim=True
                            )
                            target_max = (
                                target_network(data.next_state)
                                .gather(1, next_actions)
                                .squeeze(1)
                            )
                            target_qvalues = (
                                data.rewards.flatten()
                                + args.gamma * target_max * (1 - data.dones.flatten())
                            )

                        # Current Q-values
                        current_qvalues = (
                            q_network(data.state).gather(1, data.actions).squeeze()
                        )

                        # Compute TD-Error
                        td_error = torch.abs(current_qvalues - target_qvalues).detach()

                        # Compute loss
                        loss = F.huber_loss(
                            current_qvalues, target_qvalues, reduction="none"
                        )
                        # Importance sampling weights is applied to loss function for each experience
                        loss = torch.mean(loss * weights.to(device))
                        losses.append(loss.item())

                        # Backpropagation
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            q_network.parameters(), max_norm=args.clip_grad_norm
                        )
                        optimizer.step()

                        # Update learning rate
                        # lr_scheduler.step()

                        network_updates += 1

                    # Update the priorities in Tree
                    replay_buffer.update_priorities(tree_idxs, td_error)

                    # Reset Network
                    q_network.reset_noise()
                    target_network.reset_noise()

                    # Update target network periodically
                    if global_step % args.target_update_frequency == 0:
                        target_network_updates += 1
                        polyak_averaging(
                            source=q_network,
                            target=target_network,
                            tau=args.tau,
                        )

                    # Logging and Evaluation
                    if global_step % args.log_interval == 0:
                        avg_loss = np.mean(losses) if losses else 0
                        current_lr = optimizer.param_groups[0]["lr"]
                        avg_reward = (
                            np.mean(episode_rewards[-20:]) if episode_rewards else 0
                        )
                        avg_episode_length = (
                            np.mean(episode_lengths[-20:]) if episode_lengths else 0
                        )

                        table_data = [
                            ["General Info", ""],
                            ["  Step", global_step],
                            ["  Learning Rate", f"{current_lr:.7f}"],
                            ["  Avg Loss", f"{avg_loss: .7f}"],
                            ["Performance Metrics", ""],
                            ["  Avg Reward (Last 20)", f"{avg_reward:.5f}"],
                            ["  Cumulative Reward", f"{np.sum(episode_rewards):.5f}"],
                            [
                                "  Max Reward",
                                f"{np.max(episode_rewards) if episode_rewards else 0:.5f}",
                            ],
                            [
                                "  Min Reward",
                                f"{np.min(episode_rewards) if episode_rewards else 0:.5f}",
                            ],
                            [
                                "  Avg. Episode Length (Last 20)",
                                f"{avg_episode_length}",
                            ],
                            ["Network Updates", ""],
                            ["  Total Network Updates", network_updates],
                            ["  Target Network Updates", target_network_updates],
                            ["Replay Buffer", ""],
                            ["  Replay Buffer Size", replay_buffer.size()],
                            ["Episode Info", ""],
                            ["  Episodes Completed", len(episode_rewards)],
                        ]

                        # Print the table
                        print(
                            "\n"
                            + tabulate(
                                table_data,
                                headers=["Category/Metric", "Value"],
                                tablefmt="grid",
                                stralign="left",
                            )
                        )

                        losses = []  # Reset loss tracking after logging

                # Save model periodically
                if args.save_model and global_step % args.model_save_frequency == 0:
                    model_path = Path(args.model_dir) / f"dqn_{global_step}.pth"
                    torch.save(q_network.state_dict(), model_path)
                    logger.debug(
                        f"Saved model at step {global_step} to {args.model_dir}"
                    )

    env.close()
