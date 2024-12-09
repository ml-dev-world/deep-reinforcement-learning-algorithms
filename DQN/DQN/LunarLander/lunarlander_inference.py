import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dataclasses import dataclass


@dataclass
class Args:
    seed: int = 1


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


# Initialize environment and device
args = Args()
set_experiment_seed(args.seed)
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env.action_space.seed(args.seed)
env = RecordVideo(
    env,
    video_folder="LunarLander-agent",
    episode_trigger=lambda x: x,
)
env = RecordEpisodeStatistics(env, buffer_length=int(1e5))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and set to evaluation mode
q_network = QNetwork(env).to(device)
q_network.load_state_dict(torch.load("saved_models/q_network.pth", map_location=device))
q_network.eval()

# Run evaluation episodes
for _ in range(4):
    state, _ = env.reset(seed=args.seed)
    done = False
    while not done:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            action = np.argmax(q_values.cpu().data.numpy())
        next_state, _, termination, truncation, _ = env.step(action)
        done = termination or truncation
        state = next_state

env.close()
