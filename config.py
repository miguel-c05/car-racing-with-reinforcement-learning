# --- ENVIRONMENT PARAMETERS ---

# Action Repeat: 
# Lower (2) is better for fine control in sliding scenarios.
ACTION_REPEAT = 2

# Training Environment Counts
# Adjust based on your CPU. 8-16 is ideal.
NUM_ENVS_LOW = 4
NUM_ENVS_MED = 8
NUM_ENVS_HIGH = 16 

# --- REWARD SHAPING WEIGHTS (Linked to customization.py) ---

# We remove complex geometric rewards and trust the default "Tile" reward (~1000 pts).
# We only add penalties to prevent the agent from falling asleep.

# --- BASE REWARD SHAPING (DO NOT MODIFY) ---
STILL_PENALTY = 0.1         # Small penalty every frame the car is stationary
STILL_SPEED_THRESHOLD = 1.0 # Speed below which the car is considered "still"
MAX_STILL_STEPS = 100       # Max steps allowed without moving before truncation
LAP_FINISH_BONUS = 100.0    # Extra bonus if it actually completes a lap (rare early on)
# -------------------------------------------
MAX_OFFROAD_STEPS = 400      # Disabled

TRUNCATION_PENALTY = 5.0          # Disabled

OFFROAD_WHEEL_PENALTY = 0.025        # Disabled

MAX_LINE_DISTANCE_REWARD = 0.25  # Disabled
LINE_DISTANCE_DECAY = 7.0        # Disabled

MAX_LINE_ANGLE_REWARD = 0.0     # Disabled
LINE_ANGLE_DECAY = 0.333       # Disabled

DRIFT_PENALTY = 0.02               # Disabled
DRIFT_THRESHOLD = 0.0             # Disabled

WIGGLE_PENALTY = 0.07             # Disabled
WIGGLE_THRESHOLD = 0.0           # Disabled

TARGET_SPEED = 50.0               # Disabled

# --- TRAINING DURATION ---

TOTAL_TIMESTEPS = 2_000_000
DEBUGGING_TIMESTEPS = 100_000

# Saving & Logging
SAVE_FREQ = 100_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 5

# Directories
LOG_DIR = "./logs/ppo_standard/"
CHECKPOINT_DIR = "./models/checkpoints/"
BEST_MODEL_DIR = "./models/best_model/"

# --- MODEL PARAMETERS (PPO) ---

# High-Performance CarRacing PPO Parameters
PPO_PARAMS = {
    "policy": "CnnPolicy",
    
    # 1. State Dependent Exploration (SDE)
    # Critical for smooth steering curves.
    "use_sde": True,
    "sde_sample_freq": 4,
    
    # 2. Optimization
    "learning_rate": 3e-4,
    "n_steps": 2048,       
    "batch_size": 64,      
    "n_epochs": 10,        
    
    # 3. Reward Processing
    "gamma": 0.99,         
    "gae_lambda": 0.95,    
    
    # 4. Stability
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    
    # 5. Entropy
    # Zero entropy because the dense tile reward provides sufficient guidance.
    "ent_coef": 0.0,
    
    "normalize_advantage": True,
    "tensorboard_log": LOG_DIR
}