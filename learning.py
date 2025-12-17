"""Training infrastructure module for reinforcement learning agent.

This module provides the core training framework for PPO agents in the CarRacing
environment. It includes custom callbacks for progress tracking and a Driver class
that manages the complete training lifecycle, including model initialization,
environment setup, checkpointing, and evaluation.

Key Features:
    - Automatic frame stacking (4 frames) and image transposition (HWC→CHW)
    - Custom progress bar with real-time reward tracking
    - Configurable checkpointing and evaluation callbacks
    - Support for phased/curriculum training with custom PPO hyperparameters
    - Proper vectorized environment handling with VecFrameStack and VecTransposeImage

Typical Usage:
    ```python
    from customization import make_vec_envs
    from learning import Driver
    
    # Create training and evaluation environments
    train_env = make_vec_envs(num_envs=4, use_additional_rewards=True)
    eval_env = make_vec_envs(num_envs=1, use_additional_rewards=False)
    
    # Initialize driver and train
    driver = Driver(train_env, eval_env=eval_env, training_steps=1_000_000)
    driver.train()
    ```
"""

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
    """Custom callback providing enhanced progress tracking during training.
    
    Displays a tqdm progress bar with real-time updates on training progress,
    mean episode reward, and best reward achieved. Updates are triggered on
    each training step and reflect statistics from the episode buffer.
    
    Attributes:
        total_timesteps (int): Total number of timesteps for the training run.
        pbar (tqdm.tqdm): The progress bar instance.
        last_mean_reward (float): Most recent mean reward from episode buffer,
            used for tracking improvements.
    
    Example:
        ```python
        callback = EnhancedProgressBarCallback(total_timesteps=1_000_000)
        model.learn(total_timesteps=1_000_000, callback=[callback])
        ```
    """
    """Custom callback providing enhanced progress tracking during training.
    
    Displays a tqdm progress bar with real-time updates on training progress,
    mean episode reward, and best reward achieved. Updates are triggered on
    each training step and reflect statistics from the episode buffer.
    
    Attributes:
        total_timesteps (int): Total number of timesteps for the training run.
        pbar (tqdm.tqdm): The progress bar instance.
        last_mean_reward (float): Most recent mean reward from episode buffer,
            used for tracking improvements.
    
    Example:
        ```python
        callback = EnhancedProgressBarCallback(total_timesteps=1_000_000)
        model.learn(total_timesteps=1_000_000, callback=[callback])
        ```
    """
    def __init__(self, total_timesteps, verbose=0):
        """Initialize the progress bar callback.
        
        Args:
            total_timesteps (int): Total number of environment steps for the
                training run. Used to configure the progress bar range.
            verbose (int, optional): Verbosity level for callback logging.
                Defaults to 0 (minimal output).
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.last_mean_reward = -np.inf

    def _on_training_start(self):
        """Initialize the progress bar when training begins.
        
        Called automatically by Stable-Baselines3 at the start of training.
        Creates the tqdm progress bar with appropriate total timesteps.
        """
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")

    def _on_step(self):
        """Update progress bar on each training step.
        
        Called after every environment step during training. Updates the progress
        bar by the number of parallel environments and refreshes the displayed
        statistics (mean reward and best reward) if new episodes have completed.
        
        Returns:
            bool: Always returns True to continue training.
        """
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
        """Close the progress bar when training completes.
        
        Called automatically by Stable-Baselines3 at the end of training.
        Ensures the progress bar is properly closed and cleaned up.
        """
        if self.pbar:
            self.pbar.close()

class Driver:
    """Main training orchestrator for PPO agents in CarRacing environment.
    
    The Driver class manages the complete lifecycle of reinforcement learning
    training, including:
    - Automatic environment preprocessing (frame stacking, image transposition)
    - PPO model initialization with configurable hyperparameters
    - Checkpoint and evaluation callback setup
    - Training execution with progress tracking
    - Model loading for evaluation or continued training
    
    Critical Implementation Details:
        - Applies VecFrameStack (4 frames) followed by VecTransposeImage (HWC→CHW)
          to both training and evaluation environments to ensure shape consistency
        - Converts physics timesteps to agent timesteps by dividing by ACTION_REPEAT
        - Supports phase-specific PPO hyperparameters for curriculum learning
        - Evaluation environment must match training environment's preprocessing
    
    Attributes:
        vec_env (VecEnv): Vectorized training environment with frame stacking and transposition.
        eval_env (VecEnv): Vectorized evaluation environment (optional).
        model (PPO): The PPO model instance from Stable-Baselines3.
        training_steps (int): Number of agent steps (not physics steps) for training.
        checkpoint_callback (CheckpointCallback): Handles periodic model checkpointing.
        eval_callback (EvalCallback): Handles periodic evaluation and best model saving.
    
    Example:
        ```python
        # Standard training
        train_env = make_vec_envs(num_envs=4, use_additional_rewards=True)
        eval_env = make_vec_envs(num_envs=1, use_additional_rewards=False)
        driver = Driver(train_env, eval_env=eval_env, training_steps=1_000_000)
        driver.train()
        
        # Phased training with custom PPO params
        phase_params = {"learning_rate": 0.0001, "ent_coef": 0.01}
        driver = Driver(train_env, custom_ppo_params=phase_params)
        driver.train(steps=500_000)
        ```
    """
    """Main training orchestrator for PPO agents in CarRacing environment.
    
    The Driver class manages the complete lifecycle of reinforcement learning
    training, including:
    - Automatic environment preprocessing (frame stacking, image transposition)
    - PPO model initialization with configurable hyperparameters
    - Checkpoint and evaluation callback setup
    - Training execution with progress tracking
    - Model loading for evaluation or continued training
    
    Critical Implementation Details:
        - Applies VecFrameStack (4 frames) followed by VecTransposeImage (HWC→CHW)
          to both training and evaluation environments to ensure shape consistency
        - Converts physics timesteps to agent timesteps by dividing by ACTION_REPEAT
        - Supports phase-specific PPO hyperparameters for curriculum learning
        - Evaluation environment must match training environment's preprocessing
    
    Attributes:
        vec_env (VecEnv): Vectorized training environment with frame stacking and transposition.
        eval_env (VecEnv): Vectorized evaluation environment (optional).
        model (PPO): The PPO model instance from Stable-Baselines3.
        training_steps (int): Number of agent steps (not physics steps) for training.
        checkpoint_callback (CheckpointCallback): Handles periodic model checkpointing.
        eval_callback (EvalCallback): Handles periodic evaluation and best model saving.
    
    Example:
        ```python
        # Standard training
        train_env = make_vec_envs(num_envs=4, use_additional_rewards=True)
        eval_env = make_vec_envs(num_envs=1, use_additional_rewards=False)
        driver = Driver(train_env, eval_env=eval_env, training_steps=1_000_000)
        driver.train()
        
        # Phased training with custom PPO params
        phase_params = {"learning_rate": 0.0001, "ent_coef": 0.01}
        driver = Driver(train_env, custom_ppo_params=phase_params)
        driver.train(steps=500_000)
        ```
    """
    def __init__(self, vec_env, eval_env=None, algorithm="PPO", training_steps=cfg.DEBUGGING_TIMESTEPS, save_freq=cfg.SAVE_FREQ, 
                 checkpoint_dir=cfg.CHECKPOINT_DIR, log_dir=cfg.LOG_DIR, 
                 best_model_dir=cfg.BEST_MODEL_DIR, eval_freq=cfg.EVAL_FREQ, n_eval_episodes=cfg.N_EVAL_EPISODES,
                 custom_ppo_params=None):
        """Initialize the Driver with environments and training configuration.
        
        Sets up the training infrastructure including environment preprocessing,
        model initialization, and callback configuration. Automatically applies
        VecFrameStack and VecTransposeImage to ensure proper observation shapes.
        
        Args:
            vec_env (VecEnv): Vectorized training environment. Will be wrapped with
                VecFrameStack and VecTransposeImage if not already applied.
            eval_env (VecEnv, optional): Vectorized evaluation environment. If provided,
                will also be wrapped with VecFrameStack and VecTransposeImage to match
                the training environment's preprocessing. Defaults to None.
            algorithm (str, optional): RL algorithm to use. Currently only "PPO" is
                supported. Defaults to "PPO".
            training_steps (int, optional): Total number of physics timesteps for training.
                Will be divided by ACTION_REPEAT to get agent steps. Defaults to
                cfg.DEBUGGING_TIMESTEPS.
            save_freq (int, optional): Frequency (in agent steps) for saving model
                checkpoints. If None, automatically set to min(training_steps // 10, 50000).
                Defaults to cfg.SAVE_FREQ.
            checkpoint_dir (str, optional): Directory path for saving model checkpoints.
                Defaults to cfg.CHECKPOINT_DIR.
            log_dir (str, optional): Directory path for TensorBoard logs.
                Defaults to cfg.LOG_DIR.
            best_model_dir (str, optional): Directory path for saving the best model
                (based on evaluation performance). Defaults to cfg.BEST_MODEL_DIR.
            eval_freq (int, optional): Frequency (in agent steps) for running evaluation.
                If None, set equal to save_freq. Defaults to cfg.EVAL_FREQ.
            n_eval_episodes (int, optional): Number of episodes to run during each
                evaluation. Defaults to cfg.N_EVAL_EPISODES.
            custom_ppo_params (dict, optional): Dictionary of PPO hyperparameters to
                override defaults from cfg.PPO_PARAMS. Useful for phase-specific
                configurations in curriculum learning. Example: {"learning_rate": 0.0001,
                "ent_coef": 0.01}. Defaults to None.
        
        Raises:
            ValueError: If algorithm is not "PPO".
        
        Note:
            - Training steps are converted from physics steps to agent steps by
              dividing by cfg.ACTION_REPEAT (typically 2).
            - Both save_freq and eval_freq are adjusted by the number of parallel
              environments when setting up callbacks.
            - All output directories are created automatically if they don't exist.
        """
        
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
            # Start with default params, then override with custom ones if provided
            self.model_params = cfg.PPO_PARAMS.copy()
            if custom_ppo_params:
                # Merge custom params (phase-specific) with defaults
                self.model_params.update(custom_ppo_params)
            
            self.model_params["env"] = self.vec_env
            self.model_params["tensorboard_log"] = self.log_dir
            self.model = PPO(**self.model_params)
        
        # Setup callbacks
        self.checkpoint_callback = None
        self.eval_callback = None
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Configure checkpoint and evaluation callbacks for training.
        
        Creates CheckpointCallback for periodic model saving and EvalCallback
        for evaluation (if eval_env is provided). Frequencies are adjusted by
        the number of parallel environments to ensure correct timestep intervals.
        
        The evaluation callback saves the best-performing model based on mean
        reward across evaluation episodes.
        """
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
        """Execute the training loop for the specified number of steps.
        
        Trains the PPO model using the configured environment and callbacks.
        Displays an enhanced progress bar with real-time reward tracking during
        training. Automatically saves checkpoints and runs evaluations based on
        the configured frequencies.
        
        Args:
            steps (int, optional): Number of agent steps to train for. If None,
                uses self.training_steps (set during initialization). Defaults to None.
            reset_num_timesteps (bool, optional): Whether to reset the timestep
                counter to zero. Set to False when continuing training from a loaded
                checkpoint. Defaults to True.
        
        Example:
            ```python
            # Initial training
            driver.train(steps=1_000_000)
            
            # Continue training from checkpoint
            driver.load_model("path/to/checkpoint.zip")
            driver.train(steps=500_000, reset_num_timesteps=False)
            ```
        
        Note:
            The steps parameter refers to agent steps, not physics steps.
            Physics steps = agent steps × ACTION_REPEAT.
        """
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
        """Load a trained PPO model from disk.
        
        Loads a saved model checkpoint (`.zip` file) and assigns it to this
        Driver instance. The model can then be used for continued training,
        evaluation, or inference.
        
        Args:
            model_path (str): Absolute or relative path to the model file
                (e.g., "models/best_model.zip" or "checkpoints/ppo_model_1000000_steps.zip").
            env (VecEnv, optional): Environment to associate with the loaded model.
                If None, uses self.vec_env (the training environment). Defaults to None.
        
        Prints:
            Confirmation message with model path and device (CPU/CUDA).
        
        Example:
            ```python
            driver = Driver(vec_env)
            driver.load_model("models/best_model.zip")
            driver.train(steps=500_000, reset_num_timesteps=False)  # Continue training
            ```
        
        Note:
            When loading a model for continued training, set reset_num_timesteps=False
            in the train() call to preserve the original timestep counter.
        """
        if env is None:
            env = self.vec_env
        
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path, env=env)
        print(f"Model loaded successfully!")
        print(f"  Device: {self.model.device}")
    
    @staticmethod
    def load_model_static(model_path, env=None):
        """Load a PPO model without creating a Driver instance (static method).
        
        Utility method for loading models when you don't need the full Driver
        infrastructure. Useful for evaluation scripts, benchmarking (compare.py),
        or quick inference tasks.
        
        Args:
            model_path (str): Absolute or relative path to the model file
                (e.g., "models/best_model.zip").
            env (VecEnv, optional): Environment to associate with the loaded model.
                If None, the model is loaded without an environment (useful for
                inspecting model architecture). Defaults to None.
        
        Returns:
            PPO: The loaded PPO model instance from Stable-Baselines3.
        
        Prints:
            Confirmation message with model path and device (CPU/CUDA).
        
        Example:
            ```python
            from learning import Driver
            from customization import make_vec_envs
            
            # Load model for evaluation
            eval_env = make_vec_envs(num_envs=1, use_additional_rewards=False)
            model = Driver.load_model_static("models/best_model.zip", env=eval_env)
            
            # Run inference
            obs = eval_env.reset()
            action, _ = model.predict(obs, deterministic=True)
            ```
        
        Note:
            This method is commonly used in compare.py for benchmarking multiple
            models without creating separate Driver instances for each.
        """
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=env)
        print(f"Model loaded successfully!")
        print(f"  Device: {model.device}")
        return model