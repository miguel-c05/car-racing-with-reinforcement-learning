"""Fine-tuning script for training without line-following rewards.

Loads a pre-trained model and continues training with line distance/angle rewards
disabled. Agent relies solely on tile completion, still penalty, and lap bonuses.

Usage:
    ```bash
    python train_no_line.py <model_path> [timesteps]
    python train_no_line.py models/phased/phase_3/best_model/best_model.zip 500000
    ```

Args:
    model_path: Path to pre-trained model checkpoint
    timesteps: Optional training steps (default: 500,000)
"""

import gymnasium as gym
from customization import make_vec_envs
from learning import Driver
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import config as cfg
import os
import sys

def main():
    """Load pre-trained model and continue training without line rewards.
    
    Disables line_distance_reward and line_angle_reward on both training and
    evaluation environments, forcing the agent to rely on native tile-based
    rewards. Useful for testing if line-following rewards are necessary.
    
    CLI Args:
        sys.argv[1]: Path to pre-trained model (required)
        sys.argv[2]: Training timesteps (optional, default: 500,000)
    
    Outputs:
        Checkpoints: ./models/no_line/checkpoints/
        Best model: ./models/no_line/best_model/
        Logs: ./logs/ppo_no_line/
    """
    
    # Check if model path provided
    if len(sys.argv) < 2:
        print("Usage: python train_no_line.py <model_path> [timesteps]")
        print("\nExample:")
        print("  python train_no_line.py models/phased/phase_3_mastery/best_model/best_model.zip 500000")
        print("\nDefault timesteps: 500,000 if not specified")
        return
    
    model_path = sys.argv[1]
    training_timesteps = int(sys.argv[2]) * cfg.ACTION_REPEAT if len(sys.argv) > 2 else 500_000 * cfg.ACTION_REPEAT
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Training WITHOUT Line Rewards")
    print(f"{'='*60}")
    print(f"Loading model: {model_path}")
    print(f"Training timesteps: {training_timesteps:,}")
    print(f"Environments: {cfg.NUM_ENVS_HIGH}")
    print(f"{'='*60}\n")
    
    # Setup directories
    log_dir = "./logs/ppo_no_line/"
    checkpoint_dir = "./models/no_line/checkpoints/"
    best_model_dir = "./models/no_line/best_model/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Create environments WITHOUT line rewards
    print("Creating environments without line-following rewards...")
    train_env = make_vec_envs(num_envs=cfg.NUM_ENVS_HIGH, use_additional_rewards=True)
    train_env.line_distance_reward = False
    train_env.line_angle_reward = False
    
    eval_env = make_vec_envs(num_envs=1, use_additional_rewards=True)
    eval_env.line_distance_reward = False
    eval_env.line_angle_reward = False
    
    # Stack frames
    if not isinstance(train_env, VecFrameStack):
        train_env = VecFrameStack(train_env, n_stack=4)
    if not isinstance(eval_env, VecFrameStack):
        eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Initialize Driver with new directories
    print("Initializing Driver...")
    driver = Driver(
        vec_env=train_env,
        eval_env=eval_env,
        training_steps=training_timesteps,
        save_freq=cfg.SAVE_FREQ,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        best_model_dir=best_model_dir,
        eval_freq=cfg.EVAL_FREQ,
        n_eval_episodes=cfg.N_EVAL_EPISODES
    )
    
    # Load the pre-trained model
    print(f"\nLoading pre-trained model from: {model_path}")
    driver.load_model(model_path)
    
    # Train without resetting timestep counter
    print("\nStarting training without line rewards...")
    print("The agent will rely on:")
    print("  - Tile completion rewards (~1000 pts per lap)")
    print("  - Still penalty (to encourage movement)")
    print("  - Lap completion bonus")
    print("  - Truncation penalties\n")
    
    driver.train(steps=training_timesteps // cfg.ACTION_REPEAT, reset_num_timesteps=False)
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best model saved to: {best_model_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
