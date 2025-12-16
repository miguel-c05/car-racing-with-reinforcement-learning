# --- ENVIRONMENT PARAMETERS ---

# Action Repeat: 
# Lower (2) is better for fine control in sliding scenarios.
ACTION_REPEAT = 2

# Training Environment Counts
# Adjust based on your CPU. 8-16 is ideal.
NUM_ENVS_LOW = 4
NUM_ENVS_MED = 8
NUM_ENVS_HIGH = 16 

# --- PHASED TRAINING CONFIGURATION ---

# Define training phases with different hyperparameters and reward weights
# Each phase has its own timesteps, reward weights, and PPO parameters
TRAINING_PHASES = [
    {
        # === PHASE 1: EXPLORATION ===
        # lenient line following, high offroad tolerance, low target speed
        "name": "phase_1_exploration",
        "timesteps": 1_500_000 * ACTION_REPEAT,
        "description": "Initial exploration - encourage movement and track following",
        
        # Reward weights for this phase
        "rewards": {
            "STILL_PENALTY": 0.1,
            "STILL_SPEED_THRESHOLD": 1.0,
            "MAX_STILL_STEPS": 100,
            "LAP_FINISH_BONUS": 100.0,
            "MAX_OFFROAD_STEPS": 400,
            "TRUNCATION_PENALTY": 5.0,
            "OFFROAD_WHEEL_PENALTY": 0.08, # Was 0.03 -> 0.08
            "MAX_LINE_DISTANCE_REWARD": 0.25,
            "LINE_DISTANCE_DECAY": 7.0,
            "LINE_SAFE_DISTANCE": 3.0, # New: starts at 3.0
            "MAX_LINE_ANGLE_REWARD": 0.5,
            "LINE_ANGLE_DECAY": 0.333,
            "LINE_SAFE_ANGLE": 0.45, # New: starts at 0.45 (25 deg)
            "DRIFT_PENALTY": 0.01,
            "DRIFT_THRESHOLD": 0.0,
            "WIGGLE_PENALTY": 0.03,
            "WIGGLE_THRESHOLD": 0.3,
            "TARGET_SPEED": 10.0, # Was 25.0 -> 10.0
        },
        
        # PPO parameters for this phase
        "ppo_params": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "ent_coef": 0.01,  # Higher entropy for exploration
        }
    },
    {
        # === PHASE 2: REFINEMENT ===
        # tighter line following, moderate offroad tolerance, moderate target speed
        "name": "phase_2_refinement",
        "timesteps": 2_000_000 * ACTION_REPEAT,
        "description": "Refinement - focus on optimal line and speed",
        
        "rewards": {
            "STILL_PENALTY": 0.1,
            "STILL_SPEED_THRESHOLD": 1.0,
            "MAX_STILL_STEPS": 100,
            "LAP_FINISH_BONUS": 100.0,
            "MAX_OFFROAD_STEPS": 400,
            "TRUNCATION_PENALTY": 5.0,
            "OFFROAD_WHEEL_PENALTY": 0.13, # Was 0.08 -> 0.13
            "MAX_LINE_DISTANCE_REWARD": 0.25,
            "LINE_DISTANCE_DECAY": 5.0,
            "LINE_SAFE_DISTANCE": 2.0, # New: starts at 2.0
            "MAX_LINE_ANGLE_REWARD": 0.5,
            "LINE_ANGLE_DECAY": 0.2,
            "LINE_SAFE_ANGLE": 0.26, # New: starts at 0.26 (15 deg)
            "DRIFT_PENALTY": 0.025,
            "DRIFT_THRESHOLD": 0.0,
            "WIGGLE_PENALTY": 0.05,
            "WIGGLE_THRESHOLD": 0.3,
            "TARGET_SPEED": 15.0, # Was 25.0 -> 15.0
        },
        
        "ppo_params": {
            "learning_rate": 2e-4,  # Lower learning rate for refinement
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "ent_coef": 0.0,  # Zero entropy for exploitation
        }
    },
    {
        # === PHASE 3: MASTERY ===
        # strict offroad tolerance, higher target speed, even less tolerant line rewards
        "name": "phase_3_mastery",
        "timesteps": 1_000_000 * ACTION_REPEAT,
        "description": "Mastery - fine-tune performance and speed",
        
        "rewards": {
            "STILL_PENALTY": 0.1,
            "STILL_SPEED_THRESHOLD": 1.0,
            "MAX_STILL_STEPS": 100,
            "LAP_FINISH_BONUS": 150.0,  # Higher bonus for completion
            "MAX_OFFROAD_STEPS": 300,   # Stricter offroad tolerance
            "TRUNCATION_PENALTY": 10.0,  # Higher penalty for failures
            "OFFROAD_WHEEL_PENALTY": 0.2, # Was 0.13 -> 0.2
            "MAX_LINE_DISTANCE_REWARD": 0.12,
            "LINE_DISTANCE_DECAY": 4.0,
            "LINE_SAFE_DISTANCE": 1.5, # New: starts at 1.5
            "MAX_LINE_ANGLE_REWARD": 0.35,
            "LINE_ANGLE_DECAY": 0.2,
            "LINE_SAFE_ANGLE": 0.09, # New: starts at 0.09 (5 deg)
            "DRIFT_PENALTY": 0.025,
            "DRIFT_THRESHOLD": 0.0,
            "WIGGLE_PENALTY": 0.09,
            "WIGGLE_THRESHOLD": 0.25,
            "TARGET_SPEED": 25.0,  # Was 40.0 -> 25.0
        },
        
        "ppo_params": {
            "learning_rate": 1e-4,  # Even lower for fine-tuning
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.15,  # Tighter clipping
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "ent_coef": 0.0,
        }
    }
]

# Current active phase (can be set programmatically)
ACTIVE_PHASE = 0  # Start with phase 0

# --- BACKWARD COMPATIBILITY: Default values (using phase 2 as baseline) ---
# These are used when not in phased training mode

# --- BASE REWARD SHAPING (DO NOT MODIFY) ---
STILL_PENALTY = 0.1         # Small penalty every frame the car is stationary
STILL_SPEED_THRESHOLD = 1.0 # Speed below which the car is considered "still"
MAX_STILL_STEPS = 100       # Max steps allowed without moving before truncation
LAP_FINISH_BONUS = 100.0    # Extra bonus if it actually completes a lap (rare early on)
# -------------------------------------------
MAX_OFFROAD_STEPS = 400      

TRUNCATION_PENALTY = 5.0          

OFFROAD_WHEEL_PENALTY = 0.04 # Was 0.025 -> 0.04       

MAX_LINE_DISTANCE_REWARD = 0.15 # Was 0.25 -> 0.15 
LINE_DISTANCE_DECAY = 7.0
LINE_SAFE_DISTANCE = 2.0  # Distance threshold for "safe" line following

MAX_LINE_ANGLE_REWARD = 0.3 # Was 0.03 -> 0.5 -> 0.3   
LINE_ANGLE_DECAY = 0.333
LINE_SAFE_ANGLE = 0.26  # Angle threshold for "safe" line following (radians)

DRIFT_PENALTY = 0.02               
DRIFT_THRESHOLD = 0.0             

WIGGLE_PENALTY = 0.05             
WIGGLE_THRESHOLD = 0.3           

TARGET_SPEED = 25.0  # Was 50.0 -> 25.0           

# --- TRAINING DURATION ---

# Total timesteps (sum of all phases or standalone)
TOTAL_TIMESTEPS = 2_000_000  # Total physics steps (adjusted by action repeat)
DEBUGGING_TIMESTEPS = 100_000

# Saving & Logging
SAVE_FREQ = 100_000
EVAL_FREQ = 50_000
N_EVAL_EPISODES = 5

# Directories
LOG_DIR = "./logs/ppo_standard/"
CHECKPOINT_DIR = "./models/checkpoints/"
BEST_MODEL_DIR = "./models/best_model/"

# --- PHASE UTILITIES ---

def get_phase_config(phase_idx):
    """
    Get the configuration for a specific training phase.
    
    Args:
        phase_idx (int): Index of the phase (0-based)
        
    Returns:
        dict: Phase configuration with rewards and ppo_params
    """
    if phase_idx < 0 or phase_idx >= len(TRAINING_PHASES):
        raise ValueError(f"Invalid phase index: {phase_idx}. Must be 0-{len(TRAINING_PHASES)-1}")
    return TRAINING_PHASES[phase_idx]

def apply_phase_config(phase_idx):
    """
    Apply a phase's configuration to the global config variables.
    This updates all reward weights and returns PPO parameters.
    
    Args:
        phase_idx (int): Index of the phase to apply
        
    Returns:
        dict: PPO parameters for the phase
    """
    global STILL_PENALTY, STILL_SPEED_THRESHOLD, MAX_STILL_STEPS, LAP_FINISH_BONUS
    global MAX_OFFROAD_STEPS, TRUNCATION_PENALTY, OFFROAD_WHEEL_PENALTY
    global MAX_LINE_DISTANCE_REWARD, LINE_DISTANCE_DECAY, LINE_SAFE_DISTANCE
    global MAX_LINE_ANGLE_REWARD, LINE_ANGLE_DECAY, LINE_SAFE_ANGLE
    global DRIFT_PENALTY, DRIFT_THRESHOLD, WIGGLE_PENALTY, WIGGLE_THRESHOLD
    global TARGET_SPEED, ACTIVE_PHASE
    
    phase = get_phase_config(phase_idx)
    rewards = phase["rewards"]
    
    # Apply reward weights
    STILL_PENALTY = rewards["STILL_PENALTY"]
    STILL_SPEED_THRESHOLD = rewards["STILL_SPEED_THRESHOLD"]
    MAX_STILL_STEPS = rewards["MAX_STILL_STEPS"]
    LAP_FINISH_BONUS = rewards["LAP_FINISH_BONUS"]
    MAX_OFFROAD_STEPS = rewards["MAX_OFFROAD_STEPS"]
    TRUNCATION_PENALTY = rewards["TRUNCATION_PENALTY"]
    OFFROAD_WHEEL_PENALTY = rewards["OFFROAD_WHEEL_PENALTY"]
    MAX_LINE_DISTANCE_REWARD = rewards["MAX_LINE_DISTANCE_REWARD"]
    LINE_DISTANCE_DECAY = rewards["LINE_DISTANCE_DECAY"]
    LINE_SAFE_DISTANCE = rewards["LINE_SAFE_DISTANCE"]
    LINE_SAFE_ANGLE = rewards["LINE_SAFE_ANGLE"]
    MAX_LINE_ANGLE_REWARD = rewards["MAX_LINE_ANGLE_REWARD"]
    LINE_ANGLE_DECAY = rewards["LINE_ANGLE_DECAY"]
    DRIFT_PENALTY = rewards["DRIFT_PENALTY"]
    DRIFT_THRESHOLD = rewards["DRIFT_THRESHOLD"]
    WIGGLE_PENALTY = rewards["WIGGLE_PENALTY"]
    WIGGLE_THRESHOLD = rewards["WIGGLE_THRESHOLD"]
    TARGET_SPEED = rewards["TARGET_SPEED"]
    
    ACTIVE_PHASE = phase_idx
    
    # Return PPO params for this phase
    return phase["ppo_params"]

def get_total_phased_timesteps():
    """Calculate total timesteps across all phases"""
    return sum(phase["timesteps"] for phase in TRAINING_PHASES)

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