import gymnasium as gym
from customization import make_vec_envs
from learning import Driver
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import config as cfg
import os

def train_phase(phase_idx, previous_model_path=None):
    """
    Train a single phase of the phased training curriculum.
    
    Args:
        phase_idx (int): Index of the phase to train (0-based)
        previous_model_path (str, optional): Path to model from previous phase
        
    Returns:
        str: Path to the best model from this phase
    """
    # Get phase configuration
    phase_config = cfg.get_phase_config(phase_idx)
    phase_name = phase_config["name"]
    phase_timesteps = phase_config["timesteps"]
    
    print(f"\n{'='*60}")
    print(f"Starting Phase {phase_idx + 1}/{len(cfg.TRAINING_PHASES)}: {phase_name}")
    print(f"Description: {phase_config['description']}")
    print(f"Timesteps: {phase_timesteps:,}")
    print(f"{'='*60}\n")
    
    # Apply phase configuration (updates global config)
    phase_ppo_params = cfg.apply_phase_config(phase_idx)
    
    # Setup directories for this phase
    phase_log_dir = f"./logs/ppo_phased/{phase_name}/"
    phase_checkpoint_dir = f"./models/phased/{phase_name}/checkpoints/"
    phase_best_model_dir = f"./models/phased/{phase_name}/best_model/"
    
    os.makedirs(phase_log_dir, exist_ok=True)
    os.makedirs(phase_checkpoint_dir, exist_ok=True)
    os.makedirs(phase_best_model_dir, exist_ok=True)
    
    # Create environments
    train_env = make_vec_envs(num_envs=cfg.NUM_ENVS_HIGH)
    eval_env = make_vec_envs(num_envs=1)
    
    # Stack frames
    if not isinstance(train_env, VecFrameStack):
        train_env = VecFrameStack(train_env, n_stack=4)
    if not isinstance(eval_env, VecFrameStack):
        eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Initialize driver (creates fresh model with default params)
    driver = Driver(
        vec_env=train_env,
        eval_env=eval_env,
        training_steps=phase_timesteps,
        save_freq=cfg.SAVE_FREQ,
        checkpoint_dir=phase_checkpoint_dir,
        log_dir=phase_log_dir,
        best_model_dir=phase_best_model_dir,
        eval_freq=cfg.EVAL_FREQ,
        n_eval_episodes=cfg.N_EVAL_EPISODES,
        custom_ppo_params=phase_ppo_params  # Pass phase-specific PPO params
    )
    
    # Load weights from previous phase if available
    if previous_model_path and os.path.exists(previous_model_path):
        print(f"Loading weights from previous phase: {previous_model_path}")
        
        # Load the previous model to get its policy network weights
        from stable_baselines3 import PPO
        previous_model = PPO.load(previous_model_path)
        
        # Copy the policy and value network parameters to the new model
        driver.model.policy.load_state_dict(previous_model.policy.state_dict())
        print(f"  Transferred policy and value network weights")
        
        # Train with reset_num_timesteps=False to continue episode counters
        print(f"\nContinuing training from previous phase with new hyperparameters...")
        driver.train(steps=phase_timesteps // cfg.ACTION_REPEAT, reset_num_timesteps=False)
    else:
        print(f"Starting fresh model for this phase")
        driver.train()
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    # Return path to best model from this phase
    best_model_path = os.path.join(phase_best_model_dir, "best_model.zip")
    print(f"\nPhase {phase_idx + 1} complete!")
    print(f"Best model saved to: {best_model_path}\n")
    
    return best_model_path

def main():
    """
    Run full phased training curriculum.
    Each phase trains with different hyperparameters and reward weights,
    continuing from the previous phase's best model.
    """
    print(f"\n{'#'*60}")
    print(f"# PHASED TRAINING CURRICULUM")
    print(f"# Total Phases: {len(cfg.TRAINING_PHASES)}")
    print(f"# Total Timesteps: {cfg.get_total_phased_timesteps():,}")
    print(f"{'#'*60}\n")
    
    # Display phase overview
    for i, phase in enumerate(cfg.TRAINING_PHASES):
        print(f"Phase {i+1}: {phase['name']}")
        print(f"  Timesteps: {phase['timesteps']:,}")
        print(f"  Description: {phase['description']}")
        print()
    
    input("Press Enter to start phased training...")
    
    # Train each phase sequentially
    previous_model = None
    for phase_idx in range(len(cfg.TRAINING_PHASES)):
        previous_model = train_phase(phase_idx, previous_model)
    
    print(f"\n{'#'*60}")
    print(f"# PHASED TRAINING COMPLETE!")
    print(f"# Final model: {previous_model}")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
