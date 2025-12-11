import cv2
import numpy as np
import os
import gymnasium as gym
from customization import CustomEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from gymnasium.wrappers import RecordVideo

TOTAL_TIMESTEPS = 250_000  # Ensure this matches your training config

# Create video directory based on training timesteps
video_subfolder = f"videos/{TOTAL_TIMESTEPS}"
os.makedirs(video_subfolder, exist_ok=True)

print(f"Videos will be saved to: {video_subfolder}")

# Load the best model
best_model_path = r"C:\car-racing-with-reinforcement-learning\models\best_model\best_model.zip"
print(f"Loading model from: {best_model_path}")
model = PPO.load(best_model_path)

# Number of episodes to record
NUM_RECORD_EPISODES = 5

# Record episodes
for episode_idx in range(NUM_RECORD_EPISODES):
    print(f"\nRecording episode {episode_idx + 1}/{NUM_RECORD_EPISODES}...")
    
    # Create environment with RecordVideo wrapper
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = CustomEnvironment(env)
    
    # Add RecordVideo wrapper
    env = RecordVideo(
        env,
        video_folder=video_subfolder,
        name_prefix=f"episode_{episode_idx + 1}",
        episode_trigger=lambda x: True  # Record this episode
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Manual Transpose and FrameStack to match training environment
    # 1. Transpose (H, W, C) -> (C, H, W)
    obs = np.transpose(obs, (2, 0, 1))
    
    # 2. Stack frames: Create initial stack by repeating the first frame
    # obs shape is (1, 84, 96), we need to stack 4 frames to get (4, 84, 96)
    stacked_obs = np.repeat(obs, 4, axis=0)
    
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    while not done and not truncated:
        # Predict action
        # Model expects (12, 84, 96)
        action, _states = model.predict(stacked_obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Update frame stack
        # 1. Transpose new observation
        obs = np.transpose(obs, (2, 0, 1))
        
        # 2. Shift stack: remove oldest frame (first channel), add new frame
        # stacked_obs is (4, 84, 96), remove first channel, add new obs (1, 84, 96)
        stacked_obs = np.concatenate([stacked_obs[1:], obs], axis=0)
        
        total_reward += reward
        step_count += 1
    
    env.close()
    
    print(f"  Steps: {step_count:4d} | Reward: {total_reward:7.2f}")

print(f"\n✓ All videos saved to: {video_subfolder}")
print(f"Total episodes recorded: {NUM_RECORD_EPISODES}")