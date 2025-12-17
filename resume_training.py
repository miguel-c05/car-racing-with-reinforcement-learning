"""Resume training script for interrupted training runs.

Automatically finds the latest valid checkpoint, validates zip integrity,
calculates remaining steps, and continues training to completion.

Usage:
    ```bash
    python resume_training.py
    ```

Note:
    Skips corrupted checkpoints (<1KB or invalid zip) and uses the latest valid one.
"""

import gymnasium as gym
from customization import make_vec_envs
from learning import Driver
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import config as cfg
import os
import glob
import re

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest valid checkpoint with zip integrity validation.
    
    Filters out corrupted checkpoints (<1KB or invalid zip) and returns
    the most recent valid checkpoint by modification time.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files.
    
    Returns:
        tuple: (checkpoint_path, steps_completed) where checkpoint_path is
            the full path to the latest valid checkpoint and steps_completed
            is the number of steps extracted from the filename.
    
    Raises:
        FileNotFoundError: If no valid checkpoints are found.
    """
    import zipfile
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ppo_model_*_steps.zip"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Filter out invalid/corrupted checkpoints by checking file size and zip validity
    valid_checkpoints = []
    for ckpt in checkpoint_files:
        file_size = os.path.getsize(ckpt)
        # Skip files smaller than 1KB (likely corrupted)
        if file_size < 1024:
            print(f"Skipping corrupted checkpoint (too small): {ckpt} (size: {file_size} bytes)")
            continue
        
        # Try to open as zip file to verify integrity
        try:
            with zipfile.ZipFile(ckpt, 'r') as zip_file:
                # Test if the zip file is valid
                zip_file.testzip()
            valid_checkpoints.append(ckpt)
        except (zipfile.BadZipFile, Exception) as e:
            print(f"Skipping corrupted checkpoint (invalid zip): {ckpt} - {e}")
    
    if not valid_checkpoints:
        raise FileNotFoundError(f"No valid checkpoint files found in {checkpoint_dir}")
    
    # Get the latest valid checkpoint by modification time
    latest_checkpoint = max(valid_checkpoints, key=os.path.getmtime)
    
    # Extract step number from filename
    match = re.search(r'ppo_model_(\d+)_steps\.zip', latest_checkpoint)
    if match:
        steps_completed = int(match.group(1))
    else:
        steps_completed = 0
    
    return latest_checkpoint, steps_completed

def main():
    """Resume training from the latest valid checkpoint.
    
    Finds the most recent valid checkpoint, calculates remaining steps to reach
    TARGET_TIMESTEPS, loads the model, and continues training with the same
    configuration. Sets reset_num_timesteps=False to preserve training statistics.
    
    Workflow:
        1. Find and validate latest checkpoint
        2. Calculate remaining steps (TARGET_TIMESTEPS - steps_completed)
        3. Create environments and load model
        4. Resume training with Driver
    
    Exits early if training is already complete or no checkpoints found.
    """
    # Target timesteps to reach
    TARGET_TIMESTEPS = cfg.TOTAL_TIMESTEPS / cfg.ACTION_REPEAT
    
    # 1. Find latest checkpoint
    print(f"Looking for checkpoints in: {cfg.CHECKPOINT_DIR}")
    try:
        checkpoint_path, steps_completed = find_latest_checkpoint(cfg.CHECKPOINT_DIR)
        print(f"Found checkpoint: {checkpoint_path}")
        print(f"Steps completed: {steps_completed:,}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("No checkpoint found. Please run train.py first.")
        return
    
    # Calculate remaining steps
    remaining_steps = TARGET_TIMESTEPS - steps_completed
    
    if remaining_steps <= 0:
        print(f"\nTraining already completed!")
        print(f"Current steps: {steps_completed:,}")
        print(f"Target steps: {TARGET_TIMESTEPS:,}")
        return
    
    print(f"\nResume Configuration:")
    print(f"  Target timesteps: {TARGET_TIMESTEPS:,}")
    print(f"  Completed: {steps_completed:,}")
    print(f"  Remaining: {remaining_steps:,}")
    print(f"  Envs: {cfg.NUM_ENVS_HIGH}")
    
    # 2. Create Envs
    print("\nCreating environments...")
    train_env = make_vec_envs(num_envs=cfg.NUM_ENVS_HIGH)
    eval_env = make_vec_envs(num_envs=1)

    # 3. Stack Frames
    if not isinstance(train_env, VecFrameStack):
        train_env = VecFrameStack(train_env, n_stack=4)
        
    if not isinstance(eval_env, VecFrameStack):
        eval_env = VecFrameStack(eval_env, n_stack=4)

    # 4. Load the checkpoint
    print(f"\nLoading model from checkpoint...")
    model = PPO.load(checkpoint_path, env=train_env)
    print(f"Model loaded successfully!")
    print(f"  Device: {model.device}")
    
    # 5. Create Driver with loaded model
    driver = Driver(
        vec_env=train_env,
        eval_env=eval_env,
        training_steps=TARGET_TIMESTEPS,
        save_freq=cfg.SAVE_FREQ,
        checkpoint_dir=cfg.CHECKPOINT_DIR,
        log_dir=cfg.LOG_DIR,
        best_model_dir=cfg.BEST_MODEL_DIR,
        eval_freq=cfg.EVAL_FREQ,
        n_eval_episodes=cfg.N_EVAL_EPISODES
    )
    
    # Replace the model with the loaded one
    driver.model = model
    
    # 6. Resume Training
    print("\nResuming training...")
    print(f"Training for {remaining_steps:,} more steps")
    print("="*50)
    
    driver.train(steps=remaining_steps, reset_num_timesteps=False)
    
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Total steps reached: {TARGET_TIMESTEPS:,}")
    
    # 7. Cleanup
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
