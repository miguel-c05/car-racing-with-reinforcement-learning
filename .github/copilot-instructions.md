# AI Coding Agent Instructions

## Project Overview
Reinforcement Learning project for autonomous racing using PPO (Proximal Policy Optimization) on Gymnasium's CarRacing-v3 environment. The codebase implements custom reward shaping (PID-inspired) and curriculum learning to train agents to follow an optimal racing line.

## Architecture

### Core Components
- **`config.py`**: **Single Source of Truth**. Contains all hyperparameters, reward weights, and `TRAINING_PHASES` for curriculum learning.
- **`customization.py`**: Custom Gymnasium wrappers.
  - `CustomEnvironment`: Implements reward shaping (optimal line distance/angle), penalties (offroad, drift), and observation preprocessing (grayscale, crop).
  - `make_vec_envs()`: **Mandatory factory function** for creating environments. Ensures consistent wrapper application.
- **`learning.py`**: Training infrastructure.
  - `Driver` class: Manages PPO model lifecycle, callbacks, and environment setup.
  - Handles the critical `VecFrameStack` → `VecTransposeImage` pipeline.
- **`train_phased.py`**: Main entry point for curriculum learning. Iterates through phases defined in `config.py`.
- **`compare.py`**: **Benchmarking tool**. Loads trained models and evaluates them on the *base* environment (no custom rewards) to measure true performance.

### Critical Data Flow
1. **Raw Observation**: 96x96x3 RGB (Gymnasium default).
2. **Preprocessing**: Crop HUD (bottom 12px) → Grayscale → 84x96x1.
3. **Stacking**: `VecFrameStack` stacks last 4 frames → 84x96x4.
4. **Transposing**: `VecTransposeImage` converts HWC to CHW → 4x84x96 (Required for PPO CNNPolicy).
5. **Action Repeat**: Actions are repeated `ACTION_REPEAT` times (default 2) in the physics engine.

## Development Workflows

### Training
```powershell
# Phased training (Recommended - Curriculum Learning)
python train_phased.py

# Standard training (Single phase)
python train.py

# Resume from checkpoint (Auto-validates zip integrity)
python resume_training.py
```

### Evaluation & Benchmarking
**Critical Step**: Custom rewards are for *training guidance*. True performance is measured by the environment's native tile-based reward.
```powershell
# Compare all 'best_model.zip' files in models/ directory
# Runs on base environment (no custom rewards) to ensure fair comparison
python compare.py --episodes 20
```

### Testing & Visualization
```powershell
# Verify physics/rendering
python test_env.py

# Manual control (WASD) - Good for understanding physics
python manual_play.py

# Record video of a trained model
python record.py
```

## Project-Specific Patterns

### Configuration & Curriculum
- **Never hardcode values**. Use `config.py`.
- **Phased Training**: Defined in `cfg.TRAINING_PHASES`. Each phase has unique:
  - `rewards`: Dict of weights (e.g., `OFFROAD_WHEEL_PENALTY`).
  - `ppo_params`: Learning rate, entropy coefficient, etc.
  - `timesteps`: Duration of phase.
- **Accessing Config**: Import as `import config as cfg`. Use `cfg.get_phase_config(i)` to load specific phase settings.

### Environment Factory
**ALWAYS** use `make_vec_envs()` from `customization.py`.
- **Training**: `make_vec_envs(..., use_additional_rewards=True)`
- **Evaluation**: `make_vec_envs(..., use_additional_rewards=False)` (Enforced in `compare.py`)

### Frame Stacking & Transposing Order
The order of wrappers is strict and handled in `learning.py` / `compare.py`:
1. `Monitor`
2. `CustomEnvironment` (Grayscale/Crop)
3. `VecFrameStack` (Stacks frames)
4. `VecTransposeImage` (HWC → CHW) **MUST BE LAST** before passing to PPO.

### Reward Shaping Philosophy
- **Auxiliary Only**: Custom rewards (line following, speed maintenance) guide the agent but are not the final metric.
- **Backwards Detection**: `dot(velocity, track_tangent) < 0` triggers massive penalty.
- **Optimal Line**: Computed via Laplacian smoothing on track tiles at reset.

## Common Pitfalls
1. **Shape Mismatches**: Usually caused by missing `VecTransposeImage` or wrong wrapper order.
2. **Evaluation Bias**: Evaluating with `use_additional_rewards=True` gives inflated scores. Always use `compare.py` or disable custom rewards for true metrics.
3. **Action Repeat**: Remember that 1 agent step = `ACTION_REPEAT` physics steps.
4. **Corrupted Checkpoints**: `resume_training.py` includes logic to validate zip files before loading. Use it.
