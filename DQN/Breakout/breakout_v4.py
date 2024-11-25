import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    ProgressBarCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42


# ===== 1. Environment Setup =====
def create_env(n_envs=4, record_video=False, video_dir="./videos/", video_freq=50_000):
    env = make_atari_env("MsPacman-v4", n_envs=n_envs, seed=SEED)
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames to capture motion

    if record_video:
        env = VecVideoRecorder(
            env,
            video_folder=video_dir,
            record_video_trigger=lambda x: x % video_freq == 0,
            video_length=200,  # Capture 200 steps per video
            name_prefix="ms_pacman",
        )
    return env


env = create_env()

# ===== 2. DQN Hyperparameters =====
dqn_params = {
    "learning_rate": 1e-4,
    "buffer_size": 200_000,  # Increased buffer size for better replay sampling
    "learning_starts": 10_000,
    "batch_size": 64,  # Larger batch size for stability
    "tau": 1.0,  # Target network update rate
    "gamma": 0.99,
    "train_freq": 4,
    "target_update_interval": 8_000,  # More frequent target updates
    "gradient_steps": 1,  # Use one gradient step per train step
    "exploration_fraction": 0.2,  # Longer exploration period
    "exploration_final_eps": 0.01,
    "exploration_initial_eps": 1.0,
    "policy_kwargs": {"net_arch": [512, 256]},  # Custom network architecture
    "verbose": 0,  # Reduce logging
    "tensorboard_log": None,  # Disable TensorBoard for less clutter
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ===== 3. Callbacks Setup =====
# Save model every 250k steps
checkpoint_callback = CheckpointCallback(
    save_freq=250_000, save_path="./models/", name_prefix="dqn_pacman"
)

# Evaluate model every 100k steps and save videos
eval_env = create_env(n_envs=1, record_video=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    log_path="./logs/",
    eval_freq=100_000,
    deterministic=True,
    render=False,
)


# Add a progress bar for visual feedback
class TQDMProgressBarCallback(ProgressBarCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(
            total=self.total_timesteps, desc="Training Progress", unit="step"
        )

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


progress_bar_callback = TQDMProgressBarCallback(total_timesteps=1_000_000)

callbacks = CallbackList([checkpoint_callback, eval_callback, progress_bar_callback])

# ===== 4. Model Initialization =====
model = DQN("CnnPolicy", env, **dqn_params)

# ===== 5. Train the DQN Agent =====
print("Training DQN Agent on MsPacman...")
model.learn(total_timesteps=1_000_000, callback=callbacks)

# Save the final model
model.save("dqn_pacman_final")
print("Training complete and model saved!")


# ===== 6. Evaluation of the Trained Agent =====
def evaluate_agent(model, env, num_episodes=10):
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=num_episodes, render=False
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# Evaluate the trained agent
print("Evaluating the trained agent...")
evaluate_agent(model, eval_env)

# Close the environments
env.close()
eval_env.close()
