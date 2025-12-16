# AI Coding Agent Instructions

## Project Overview
Reinforcement Learning project for autonomous racing using PPO (Proximal Policy Optimization) on Gymnasium's CarRacing-v3 environment. The codebase implements custom reward shaping inspired by PID controllers and racing line optimization.

## Architecture

### Core Components
- **`config.py`**: Central configuration hub for hyperparameters, reward weights, and training settings. All tunable parameters live here.
- **`customization.py`**: Custom Gymnasium wrappers that implement reward shaping and observation preprocessing
  - `CustomEnvironment`: Adds PID-inspired rewards (optimal line distance/angle), penalties (offroad, drift, wiggle), and processes observations (crops HUD, converts to grayscale)
  - `CustomRepeatWrapper`: Action repeat mechanism (default 2x) to reduce temporal resolution
  - `make_vec_envs()`: Factory function for creating vectorized environments with SubprocVecEnv
- **`learning.py`**: Training infrastructure
  - `Driver` class: Manages PPO model lifecycle, handles frame stacking/transposing, coordinates callbacks
  - Applies `VecTransposeImage` wrapper to ensure (C,H,W) format expected by CNNPolicy
- **`train.py`**: Main training entry point - creates environments, initializes Driver, runs training
- **`resume_training.py`**: Checkpoint recovery with validation (checks zip integrity, file size)

### Critical Data Flow
1. Raw observation (96x96x3 RGB) → crop HUD bottom 12px → grayscale (84x96x1)
2. Single frame → stack 4 frames (84x96x4) via `VecFrameStack`
3. Transpose (H,W,C) → (C,H,W) via `VecTransposeImage` for CNN policy
4. Action repeat: Each agent action executed 2x in physics simulation (configured via `ACTION_REPEAT`)

### Reward Shaping Philosophy
The environment's native tile-based reward (~1000 pts for full lap) is trusted as the primary signal. Additional rewards/penalties are **auxiliary guidance only**:
- **Optimal line rewards**: Laplacian-smoothed racing line computed at reset. Rewards based on distance to line and velocity angle alignment (PID-inspired: position error = P, velocity angle = D)
- **Backwards detection**: Heavy penalty if velocity opposes track tangent direction (prevents exploiting tile rewards by reversing)
- **Penalties**: Still (< 1.0 speed), offroad wheels, drift (lateral velocity), wiggle (steering oscillation)
- All weights configured in `config.py` (e.g., `MAX_LINE_DISTANCE_REWARD`, `DRIFT_PENALTY`)

## Development Workflows

### Training
```powershell
# Standard training (2M timesteps default)
python train.py

# Phased training with curriculum learning (3 phases)
python train_phased.py

# Resume from latest checkpoint (auto-validates integrity)
python resume_training.py
```

**Phased Training:**
- Implements curriculum learning with 3 configurable phases (exploration → refinement → mastery)
- Each phase has distinct reward weights, PPO hyperparameters, and timestep allocations
- Phases defined in `config.py` under `TRAINING_PHASES` list
- Each phase continues from previous phase's best model
- Models saved to `models/phased/<phase_name>/`
- Logs to `logs/ppo_phased/<phase_name>/`

**Standard Training:**
- Models saved to `models/` with subdirectories: `checkpoints/`, `best_model/`
- Logs to `logs/ppo_standard/` (TensorBoard format)
- Evaluation runs every `EVAL_FREQ` steps, saves best model based on mean reward

### Testing & Visualization
```powershell
# Verify Box2D physics engine
python test_env.py

# Manual control (WASD keys)
python manual_play.py

# Generate videos from trained model
python record.py  # Outputs to videos/<timesteps>/
```

### Hyperparameter Tuning
Modify `config.py`, then create new log/model directories:
- Change `LOG_DIR` to avoid overwriting (e.g., `logs/ppo_standard/v6_new_experiment/`)
- Adjust reward weights (`OFFROAD_WHEEL_PENALTY`, `LINE_ANGLE_DECAY`, etc.)
- PPO params in `PPO_PARAMS` dict (learning_rate, n_steps, entropy, etc.)

## Project-Specific Patterns

### Configuration Management
**Never hardcode hyperparameters.** All tunable values in `config.py` with descriptive comments. Import as `import config as cfg` and reference `cfg.PARAMETER_NAME`.

**Phased Training Config:**
- Use `cfg.get_phase_config(phase_idx)` to retrieve phase configuration
- Use `cfg.apply_phase_config(phase_idx)` to apply phase settings globally (updates all reward weights and returns PPO params)
- Access current active phase via `cfg.ACTIVE_PHASE`
- Each phase in `TRAINING_PHASES` has: `name`, `timesteps`, `description`, `rewards` dict, `ppo_params` dict

### Environment Creation
Always use `make_vec_envs()` from `customization.py`, not raw `gym.make()`. This ensures consistent wrapper application (Monitor → CustomEnvironment → CustomRepeatWrapper → SubprocVecEnv).

### Frame Stacking & Transposing
**Critical order:**
1. `VecFrameStack` first (stacks last 4 grayscale frames)
2. `VecTransposeImage` second (HWC → CHW for CNN)

Missing or wrong order causes shape mismatches. Driver class handles this automatically for both train and eval envs.

### Checkpoint Management
`resume_training.py` validates checkpoints before loading (zip integrity, >1KB size). Corrupted checkpoints are auto-skipped to find latest valid one.

### Observation Space
Output: `(84, 96, 1)` grayscale, single channel. After stacking: `(84, 96, 4)`. After transpose: `(4, 84, 96)` or `(12, 84, 96)` depending on context (see `VecTransposeImage` docs).

## Key Implementation Details

### Optimal Racing Line
Computed in `CustomEnvironment.get_optimal_line()` using Laplacian smoothing (200 iterations) on track tile centers. Line stays within `TRACK_WIDTH - 1.5` margin. Recalculated every episode reset.

### Terminal Observation Handling
`CustomEnvironment.step()` must process `info["terminal_observation"]` to match observation space (crop HUD, grayscale, add channel dim). SB3 uses this for bootstrap value estimation on truncation.

### Backwards Driving Detection
`get_line_distance_and_angle_diff()` returns 4 values: `line_distance`, `angle_diff`, `closest_idx`, `is_backwards`. Backwards flag based on `dot(car_velocity, track_tangent) < 0`. Used to apply heavy penalty and prevent reverse-driving exploits.

### SDE (State-Dependent Exploration)
Enabled in `PPO_PARAMS` with `use_sde=True, sde_sample_freq=4`. Critical for smooth steering curves in continuous action space. Do not disable.

## Environment Setup
Conda environment defined in `environment.yml`:
```powershell
conda env create -f environment.yml
conda activate rl-racing
```
Key dependencies: `gymnasium`, `stable-baselines3`, `box2d-py`, `opencv`, `pygame`

## Directory Structure
- `models/`: Saved models, organized by version (e.g., `re_v5/best_model/`, `re_v5/checkpoints/`)
- `logs/`: TensorBoard logs (view with `tensorboard --logdir logs/`)
- `videos/`: Recorded episodes from `record.py`
- `depreciated/`: Old model versions and videos for reference

## Common Pitfalls
1. **Shape mismatch errors**: Verify frame stacking/transposing order in Driver class
2. **Corrupted checkpoints**: Use `resume_training.py`, not direct `PPO.load()` on untrusted checkpoints
3. **Forgetting ACTION_REPEAT**: Training timesteps are physics steps × ACTION_REPEAT, not agent steps
4. **Eval env shape mismatch**: Eval env must match train env wrappers exactly (both need VecTransposeImage)
5. **Modifying reward without updating config**: Keep `config.py` as single source of truth

## Monitoring Training
TensorBoard metrics in `logs/ppo_standard/<experiment_name>/`:
- `rollout/ep_rew_mean`: Primary metric (target ~900+ for lap completion)
- `train/entropy_loss`: Should stay near 0.0 (entropy disabled via `ent_coef=0.0`)
- Eval callback writes to `logs/ppo_standard/eval/evaluations.npz`
