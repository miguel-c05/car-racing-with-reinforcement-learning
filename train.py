import gymnasium as gym
from customization import make_vec_envs
from learning import Driver
from stable_baselines3.common.vec_env import VecFrameStack
import config as cfg

def main():
    cfg.TOTAL_TIMESTEPS = 4_000_000  # 8 million timesteps for final training
    cfg.SAVE_FREQ = 500
    # 1. Setup
    print(f"Configuration:")
    print(f"  Envs: {cfg.NUM_ENVS_HIGH}")
    print(f"  Steps: {cfg.TOTAL_TIMESTEPS:,}")
    
    # 2. Create Envs
    train_env = make_vec_envs(num_envs=cfg.NUM_ENVS_HIGH)
    
    # Use standard make_vec_envs for eval to ensure consistency
    eval_env = make_vec_envs(num_envs=1)

    # 3. Stack Frames
    # Input: (84, 96, 1) -> Output: (84, 96, 4)
    # SB3 handles the (H, W, C) -> (C, H, W) transpose internally.
    if not isinstance(train_env, VecFrameStack):
        train_env = VecFrameStack(train_env, n_stack=4)
        
    if not isinstance(eval_env, VecFrameStack):
        eval_env = VecFrameStack(eval_env, n_stack=4)

    # 4. Initialize Driver
    driver = Driver(
        vec_env=train_env,
        eval_env=eval_env,
        training_steps=cfg.TOTAL_TIMESTEPS,
        save_freq=cfg.SAVE_FREQ,
        checkpoint_dir=cfg.CHECKPOINT_DIR,
        log_dir=cfg.LOG_DIR,
        best_model_dir=cfg.BEST_MODEL_DIR,
        eval_freq=cfg.EVAL_FREQ,
        n_eval_episodes=cfg.N_EVAL_EPISODES
    )

    # 5. Train
    print("Starting training...")
    driver.train()
    
    # 6. Cleanup
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()