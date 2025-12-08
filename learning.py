import numpy as np
import stable_baselines3 as sb3
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import cv2
import os
import glob
import re

class Driver:
    def __init__(self, vec_env, eval_env=None, algorithm="PPO", training_steps=100000, save_freq=None, 
                 checkpoint_dir="./mod_models/checkpoints/", log_dir="./mod_models/logs/", 
                 best_model_dir="./mod_models/best_model/", eval_freq=None, n_eval_episodes=5):
        self.vec_env = vec_env
        self.eval_env = eval_env
        self.training_steps = training_steps
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.best_model_dir = best_model_dir
        
        MAXIMUM_SAVE_FREQ = 50000
        
        if self.save_freq is None:
            self.save_freq = min(training_steps // 10, MAXIMUM_SAVE_FREQ)
        
        if self.eval_freq is None:
            self.eval_freq = self.save_freq

        # Stack 4 frames so the agent can perceive velocity and acceleration
        # If vec_env is not already stacked, we wrap it here.
        if not isinstance(vec_env, VecFrameStack):
            self.vec_env = VecFrameStack(vec_env, n_stack=4)
            
        self.algorithm = algorithm.lower()
        
        # Create directories for saving models and logs
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        
        # --- HYPERPARAMETER CONFIGURATION ---
        self.model_params = {
            "policy": "CnnPolicy",
            "env": self.vec_env,
            
            # 1. State Dependent Exploration (SDE)
            # Critical for smooth steering (avoids jittery "bang-bang" control)
            "use_sde": True,
            "sde_sample_freq": 4,
            
            # 2. Learning Rate
            # Start slightly lower than default to prevent early collapse
            "learning_rate": 3e-4, 
            
            # 3. Batch & Buffer Sizes
            # Large buffer to capture long-term consequences of drifting/braking
            "n_steps": 2048,
            "batch_size": 128,
            "n_epochs": 10,
            
            # 4. Discount Factor (Gamma)
            # High gamma because current actions (steering) affect the car 
            # far into the future (making the turn)
            "gamma": 0.99,
            "gae_lambda": 0.95,
            
            # 5. Clipping & Entropy
            # Clip range standard (0.2). 
            # Entropy (0.01) prevents the agent from deciding too early to just "park"
            "clip_range": 0.2,
            "ent_coef": 0.01,
            
            # 6. TensorBoard logging
            "tensorboard_log": log_dir,
        }

        if self.algorithm == "ppo":
            self.model = PPO(**self.model_params)
        
        # Setup callbacks
        self.checkpoint_callback = None
        self.eval_callback = None
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup training callbacks for checkpointing and evaluation"""
        num_envs = self.vec_env.num_envs
        
        # Checkpoint callback
        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq // num_envs,
            save_path=self.checkpoint_dir,
            name_prefix="ppo_model"
        )
        
        # Evaluation callback
        if self.eval_env is not None:
            self.eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=self.best_model_dir,
                log_path=os.path.join(self.log_dir, "eval/modified_model"),
                eval_freq=max(self.eval_freq // num_envs, 1),
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                render=False
            )
        
    def train(self, steps=None, reset_num_timesteps=True):
        """Train the model for specified steps"""
        if steps is None:
            steps = self.training_steps
            
        callbacks = [self.checkpoint_callback]
        if self.eval_callback is not None:
            callbacks.append(self.eval_callback)
            
        self.model.learn(
            total_timesteps=steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps
        )
    
    def resume_training(self, target_steps=None, num_envs=None):
        """
        Resume training from the latest checkpoint.
        
        Args:
            target_steps: Total target steps to reach (default: self.training_steps)
            num_envs: Number of environments for resumed training (default: keep same)
        """
        if target_steps is None:
            target_steps = self.training_steps
            
        # Find the latest checkpoint
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "ppo_model_*_steps.zip"))
        
        if not checkpoint_files:
            print("No checkpoint files found. Starting training from scratch...")
            self.train(steps=target_steps)
            return
        
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        # Extract step number from checkpoint filename
        match = re.search(r'ppo_model_(\d+)_steps\.zip', latest_checkpoint)
        if not match:
            print("Could not extract step number from checkpoint filename.")
            return
            
        current_steps = int(match.group(1))
        print(f"Checkpoint is at step: {current_steps}")
        
        # Calculate remaining steps
        remaining_steps = target_steps - current_steps
        
        if remaining_steps <= 0:
            print(f"Training already completed! Current steps ({current_steps}) >= Target ({target_steps})")
            return
        
        print(f"Remaining steps to train: {remaining_steps}")
        
        # Create new environment if num_envs is specified
        if num_envs is not None and num_envs != self.vec_env.num_envs:
            print(f"Creating new environment with {num_envs} workers...")
            # This assumes you're using CarRacing-v3, adjust if needed
            new_vec_env = make_vec_env("CarRacing-v3", n_envs=num_envs)
            new_vec_env = VecFrameStack(new_vec_env, n_stack=4)
            
            # Load model with new environment
            self.model = PPO.load(latest_checkpoint, env=new_vec_env)
            self.vec_env = new_vec_env
            
            # Update callbacks
            self._setup_callbacks()
        else:
            # Load model with existing environment
            self.model = PPO.load(latest_checkpoint, env=self.vec_env)
        
        print(f"Model loaded successfully. Resuming training...")
        
        # Train for remaining steps
        self.train(steps=remaining_steps, reset_num_timesteps=False)
        
        print(f"Training completed! Total steps reached: {target_steps}")
        
    def save(self, path):
        """Save the model to specified path"""
        self.model.save(path)
        print(f"Model saved to: {path}")


# --- UPDATED LEARN/VISUALIZE FUNCTION ---
def learn(driver : Driver, num_envs : int = 2, visualize=True):
    # Use the driver's vec_env (which might be FrameStacked now)
    vec_env = driver.vec_env
    
    # Reset everything to start
    obs = vec_env.reset()

    try:
        while True:
            # 1. PREDICT ACTIONS using the trained model
            # deterministic=True gives the best behavior, False gives exploration
            actions, _states = driver.model.predict(obs, deterministic=True)
            
            # 2. Step all environments simultaneously
            obs, rewards, dones, infos = vec_env.step(actions)
            
            if visualize:
                n_envs = vec_env.num_envs
                
                # Dynamic Grid Calculation
                grid_cols = int(np.ceil(np.sqrt(n_envs)))
                grid_rows = int(np.ceil(n_envs / grid_cols))
                
                rows = []
                for r in range(grid_rows):
                    row_images = []
                    for c in range(grid_cols):
                        idx = r * grid_cols + c
                        if idx < n_envs:
                            # OBS HANDLING FOR FRAMESTACK
                            # VecFrameStack returns (96, 96, 3*n_stack) = (96, 96, 12)
                            # We only want to visualize the current frame (the last 3 channels)
                            current_frame = obs[idx][:, :, -3:]
                            row_images.append(current_frame)
                        else:
                            # Pad with black image
                            row_images.append(np.zeros((96, 96, 3), dtype=np.uint8))
                    
                    rows.append(np.hstack(row_images))
                
                grid_image = np.vstack(rows)
                grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
                grid_image_big = cv2.resize(grid_image_bgr, (800, 800), interpolation=cv2.INTER_NEAREST)
    
                cv2.imshow(f"Agent View ({n_envs}x)", grid_image_big)
    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        vec_env.close()
        cv2.destroyAllWindows()