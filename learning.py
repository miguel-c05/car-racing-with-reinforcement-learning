import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from tqdm.auto import tqdm
import os
import config as cfg

# --- Custom Progress Bar ---
class EnhancedProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_mean_reward = -np.inf

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([info['r'] for info in self.model.ep_info_buffer])
            
            if mean_reward != self.last_mean_reward:
                self.pbar.set_postfix({
                    "Mean Reward": f"{mean_reward:.1f}",
                    "Best": f"{self.last_mean_reward:.1f}"
                })
                self.last_mean_reward = mean_reward
            
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

class Driver:
    def __init__(self, vec_env, eval_env=None, algorithm="PPO", training_steps=cfg.DEBUGGING_TIMESTEPS, save_freq=cfg.SAVE_FREQ, 
                 checkpoint_dir=cfg.CHECKPOINT_DIR, log_dir=cfg.LOG_DIR, 
                 best_model_dir=cfg.BEST_MODEL_DIR, eval_freq=cfg.EVAL_FREQ, n_eval_episodes=cfg.N_EVAL_EPISODES):
        
        self.vec_env = vec_env
        self.eval_env = eval_env
        
        # Calculate Agent Steps from Physics Steps
        self.training_steps = training_steps // cfg.ACTION_REPEAT
        
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

        # Stack 4 frames for Training Env
        if not isinstance(vec_env, VecFrameStack):
            self.vec_env = VecFrameStack(vec_env, n_stack=4)
        
        # FIX: Explicitly Transpose Training Env to ensure consistent shape (C, H, W)
        # This prevents PPO from doing it implicitly and prevents the (6, 84, 96) shape error
        if not isinstance(self.vec_env, VecTransposeImage):
            self.vec_env = VecTransposeImage(self.vec_env)
            
        # FIX: Ensure Eval Env is stacked AND transposed to match Training Env
        if self.eval_env is not None:
            if not isinstance(self.eval_env, VecFrameStack):
                self.eval_env = VecFrameStack(self.eval_env, n_stack=4)
            
            # CRITICAL: PPO auto-wraps training env in VecTransposeImage (HWC->CHW).
            # We must manually wrap eval_env to match, otherwise we get a type warning
            # and the model receives scrambled shapes during eval.
            if not isinstance(self.eval_env, VecTransposeImage):
                self.eval_env = VecTransposeImage(self.eval_env)
            
        self.algorithm = algorithm.lower()
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        
        # --- MODEL SETUP ---
        if self.algorithm == "ppo":
            self.model_params = cfg.PPO_PARAMS
            self.model_params["env"] = self.vec_env
            self.model_params["tensorboard_log"] = self.log_dir
            self.model = PPO(**self.model_params)
        
        # Setup callbacks
        self.checkpoint_callback = None
        self.eval_callback = None
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        num_envs = self.vec_env.num_envs
        self.checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq // num_envs,
            save_path=self.checkpoint_dir,
            name_prefix="ppo_model"
        )
        if self.eval_env is not None:
            self.eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=self.best_model_dir,
                log_path=os.path.join(self.log_dir, "eval"),
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
        
        # Add Progress Bar
        progress_callback = EnhancedProgressBarCallback(total_timesteps=steps)
        callbacks.append(progress_callback)
            
        self.model.learn(
            total_timesteps=steps,
            callback=callbacks,
            progress_bar=False, 
            reset_num_timesteps=reset_num_timesteps
        )
    
    def load_model(self, model_path, env=None):
        if env is None:
            env = self.vec_env
        
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path, env=env)
        print(f"Model loaded successfully!")
        print(f"  Device: {self.model.device}")
    
    @staticmethod
    def load_model_static(model_path, env=None):
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env)
        print(f"Model loaded successfully!")
        print(f"  Device: {model.device}")
        return model