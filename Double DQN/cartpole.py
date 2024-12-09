import flappy_bird_gymnasium

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
    cuda: bool = True

    # Environment Settings
    env_id: str = "CartPole-v1"

    # Exploration Settings
    start_epsilon: float = 1.0
    end_epsilon: float = 0.01  # Better exploration-exploitation tradeoff
    decay_rate: int = 200  # Faster epsilon decay for smaller environments
    exploration_fraction: int = 0.5

    # Training Settings
    total_timesteps: int = 10000
    buffer_size: int = 100000  # Reduced for CartPole
    gamma: float = 0.99  # Standard for shorter episodes
    tau: float = 1e-2  # Smooth target updates
    train_frequency: int = 2  # Train less often
    target_update_frequency: int = 1  # Update target less frequently
    learning_start: int = 100  # Start training earlier

    # Model settings
    batch_size: int = 64
    learning_rate: float = 2e-4
    clip_grad_norm: int = 10
    weight_decay: int = 0
    gradient_steps: int = 1

    # Saving & Logging Settings
    capture_video: bool = True
    video_dir: str = "runs/videos"
    save_model: bool = True
    model_dir: str = "runs/models"
    model_save_frequency: int = 5000
    log_interval: int = 500  # Log more frequently


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
    # Ensure video directory exists
    os.makedirs(f"videos/{run_name}", exist_ok=True)

    # Create environment with appropriate render mode
    env = gym.make(
        env_id,
        render_mode="rgb_array",
        # use_lidar=False,
    )

    # Add RecordVideo wrapper if capture_video is enabled
    if capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=f"videos/{run_name}",
            episode_trigger=lambda x: x % 500 == 0,  # Save video for every episode
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)  # Optionally track stats

    # Set seed for reproducibility
    env.action_space.seed(seed)
    return env


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
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
        self.full = False

    def add(self, state, next_state, action, reward, done, info):
        self.states[self.pos] = state.clone()
        self.next_states[self.pos] = next_state.clone()
        self.actions[self.pos] = action.unsqueeze(-1).clone()
        self.rewards[self.pos] = reward.clone()
        self.dones[self.pos] = done.clone()
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
        state = self.states[batch_inds]
        next_state = self.next_states[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds].view(-1, 1)
        dones = self.dones[batch_inds].int()
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
    def __init__(self, env: gym.Env, linear1_units=512, linear2_units=256):
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


def exponential_schedule(start_e: float, end_e: float, decay_rate: float, t: int):
    return end_e + (start_e - end_e) * np.exp(-t / decay_rate)


def linear_schedule(
    start_e: float, end_e: float, duration: int, t: int, eps_decay: float = 0.995
):
    return max(end_e, start_e - (start_e - end_e) * (t / duration))


def soft_update(source, target, tau):
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
    q_network = QNetwork(env=env).to(device)
    target_network = QNetwork(env=env).to(device)
    target_network.load_state_dict(
        q_network.state_dict()
    )  # Sync target network with Q-network
    return q_network, target_network


def initialize_replay_buffer(buffer_size, observation_space_shape, device):
    return ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=observation_space_shape,
        action_space=1,  # Assuming discrete action space with 1 action dimension
        device=device,
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

    # Replay buffer setup
    logger.info(f"Initializing replay buffer with buffer size: {args.buffer_size}...")
    replay_buffer = initialize_replay_buffer(
        buffer_size=args.buffer_size,
        observation_space_shape=env.observation_space.shape,
        device=device,
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
                # Update epsilon using exponential schedule
                # epsilon = exponential_schedule(
                #     start_e=args.start_epsilon,
                #     end_e=args.end_epsilon,
                #     decay_rate=args.decay_rate,
                #     t=global_step,
                # )
                # Update epsilon using linear schedule
                epsilon = linear_schedule(
                    args.start_epsilon,
                    args.end_epsilon,
                    args.exploration_fraction * args.total_timesteps,
                    global_step,
                )

                # Initialize stats for the current episode
                current_episode_reward = 0
                current_episode_length = 0

                # Reset environment state
                state, _ = env.reset(seed=args.seed)
                state = torch.tensor(state, dtype=torch.float32, device=device)

                while True:
                    # Choose action: Exploration vs. Exploitation
                    if np.random.random() < epsilon:
                        action = env.action_space.sample()
                        exploration_actions += 1
                    else:
                        # print(state.shape)
                        q_values = q_network(state.unsqueeze(0))
                        action = torch.argmax(q_values).cpu().numpy()
                        exploitation_actions += 1

                    # Take a step in the environment
                    next_state, reward, termination, truncation, info = env.step(action)
                    done = termination or truncation
                    real_next_state = next_state.copy()

                    # Handle truncation for final observations
                    if "final_observation" in info:
                        real_next_state = info["final_observation"]

                    # Clip rewards for stability
                    reward = np.clip(reward, -1, 1)

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
                        data = replay_buffer.sample(args.batch_size)

                        # Double Q-Learning target calculation
                        with torch.no_grad():
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

                        # Compute loss
                        loss = F.mse_loss(target_qvalues, current_qvalues)
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

                    # Update target network periodically
                    if global_step % args.target_update_frequency == 0:
                        target_network_updates += 1
                        soft_update(
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
                            ["  Epsilon", f"{epsilon:.4f}"],
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
                            ["Action Statistics", ""],
                            ["  Exploration Actions", exploration_actions],
                            ["  Exploitation Actions", exploitation_actions],
                            [
                                "  Exploration Ratio",
                                f"{exploration_actions / (exploration_actions + exploitation_actions):.2%}",
                            ],
                            ["Episode Info", ""],
                            ["  Episodes Completed", len(episode_rewards)],
                        ]

                        # Add a blank line before each category for readability
                        formatted_table_data = []
                        for row in table_data:
                            # if row[1] == "":  # Category header
                            #     formatted_table_data.append(
                            #         ["", ""]
                            #     )  # Blank line before category
                            formatted_table_data.append(row)

                        # Print the table
                        print(
                            "\n"
                            + tabulate(
                                formatted_table_data,
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
