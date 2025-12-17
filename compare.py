"""Model benchmarking tool for comparing trained PPO agents.

Evaluates multiple models on base CarRacing-v3 environment (no custom rewards)
to measure true agent performance. Recursively discovers all best_model.zip files
and outputs per-episode scores to NumPy arrays with summary statistics.

Usage:
    ```bash
    python compare.py --episodes 20
    python compare.py --models_dir ./trained_models
    ```

Key Design:
    - use_additional_rewards=False for fair comparison on native rewards
    - Deterministic evaluation for reproducibility
    - Matches training preprocessing (VecFrameStack → VecTransposeImage)
    - Saves results to .npy files for efficient numerical analysis
"""

import argparse
import os
import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from tqdm import tqdm

# Import local modules to ensure environment matches training exactly
from customization import make_vec_envs
import config as cfg

def setup_eval_env(num_envs=1):
    """Create evaluation environment matching training preprocessing.
    
    Applies same pipeline as learning.py (grayscale, frame stacking, transposition)
    but with use_additional_rewards=False for native reward evaluation.
    
    Args:
        num_envs (int): Number of parallel environments. Defaults to 1.
    
    Returns:
        VecTransposeImage: Configured environment with shape (num_envs, 4, 84, 96).
    """
    # 1. Create Base Env (Custom wrappers for Grayscale/Crop, but Standard Rewards)
    env = make_vec_envs(num_envs=num_envs, use_additional_rewards=False)
    
    # 2. Frame Stack (Match Training)
    env = VecFrameStack(env, n_stack=4)
    
    # 3. Transpose (Match Training - Critical fix from learning.py)
    env = VecTransposeImage(env)
    
    return env

def evaluate_model(model_path, env, n_episodes=5, model_name="model"):
    """Load and evaluate a trained PPO model for multiple episodes.
    
    Uses deterministic action selection and returns episode rewards.
    
    Args:
        model_path (str): Path to model zip file.
        env (VecEnv): Environment with matching preprocessing pipeline.
        n_episodes (int): Number of episodes to evaluate. Defaults to 5.
        model_name (str): Model identifier for progress bar. Defaults to "model".
    
    Returns:
        list[float]: Total reward for each episode. Empty list if loading fails.
    """
    try:
        # Load the model
        # We don't need the Driver class here, just the loaded PPO object
        model = PPO.load(model_path, env=env, print_system_info=False)
    except Exception as e:
        print(f"\nError loading {model_path}: {e}")
        return []

    episode_rewards = []
    
    # Progress bar for episodes
    episode_pbar = tqdm(range(n_episodes), desc=f"  Episodes for {model_name}", leave=False)
    for _ in episode_pbar:
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        # Determine if we need to reset states (for LSTM policies, though we use CNN)
        # For standard CNN PPO, state is None
        state = None
        
        while not done:
            # Predict action (Deterministic=True is standard for evaluation)
            action, state = model.predict(obs, state=state, deterministic=True)
            
            # Step env
            obs, rewards, dones, infos = env.step(action)
            
            # Sum reward (VecEnv returns array of rewards)
            total_reward += rewards[0]
            done = dones[0]
            
        episode_rewards.append(total_reward)
        episode_pbar.set_postfix({"last_score": f"{total_reward:.1f}"})
        
    return episode_rewards

def main():
    """Main execution function for model comparison.
    
    Recursively discovers best_model.zip files, evaluates each for specified
    episodes, saves results to NumPy arrays, and displays summary statistics.
    
    CLI Args:
        --models_dir: Root directory for models. Default: "./models"
        --episodes: Episodes per model. Default: 20
    
    Output:
        - comparison_results.npy: Array (N, 3) with [Model Name, Episode, Score]
        - comparison_scores.npy: Array (N,) with scores only
        - Console summary with mean, std, max per model
    """
    parser = argparse.ArgumentParser(description="Compare PPO models on the base CarRacing environment.")
    
    # Defaults based on your config.py
    parser.add_argument("--models_dir", type=str, default="./models", 
                        help="Root models directory containing version folders")
    parser.add_argument("--episodes", type=int, default=20, 
                        help="Number of episodes to run per model")
    
    args = parser.parse_args()

    # 1. Find Models - Recursively search for best_model.zip files
    model_files = []
    
    # Use glob to recursively find all best_model.zip files
    pattern = os.path.join(args.models_dir, "**", "best_model", "best_model.zip")
    model_files = glob.glob(pattern, recursive=True)
         
    if not model_files:
        print(f"No best_model.zip files found in {args.models_dir}")
        print(f"Expected structure: models/{{version}}/best_model/best_model.zip")
        return

    # Sort files by path for consistent ordering
    model_files.sort()
    
    print(f"Found {len(model_files)} models in {args.models_dir}")
    print("Model paths found:")
    for mf in model_files:
        print(f"  {mf}")
    print(f"\nEvaluating each for {args.episodes} episodes on Base Environment (Standard Rewards)...")

    # 2. Setup Environment
    # We create ONE env and reuse it.
    env = setup_eval_env(num_envs=1)

    # 3. Evaluation Loop
    all_results = []  # Will store tuples of (model_name, episode, score)

    pbar = tqdm(model_files, desc="Evaluating Models", position=0)
    for model_path in pbar:
        # Extract meaningful name from path
        # Normalize path separators for consistent processing
        normalized_path = model_path.replace(os.sep, '/')
        path_parts = normalized_path.split('/')
        
        # Find the models directory index
        models_idx = None
        for i, part in enumerate(path_parts):
            if part == "models":
                models_idx = i
                break
        
        if models_idx is not None and models_idx + 1 < len(path_parts):
            # Get version (e.g., "re_v5", "re_v6", "no_line")
            version = path_parts[models_idx + 1]
            
            # Check if there's a phase directory
            if "phase_" in normalized_path:
                # Find the phase directory
                for part in path_parts[models_idx + 2:]:
                    if part.startswith("phase_"):
                        model_name = f"{version}_{part}"
                        break
                else:
                    model_name = version
            else:
                model_name = version
        else:
            model_name = "unknown"
        
        pbar.set_description(f"Evaluating {model_name}")
        
        rewards = evaluate_model(model_path, env, args.episodes, model_name)
        
        for episode_num, score in enumerate(rewards, start=1):
            all_results.append([model_name, episode_num, score])

    # 4. Cleanup
    env.close()
    
    # 5. Convert to NumPy arrays and save
    if all_results:
        # Full results array: [Model Name, Episode, Score]
        results_array = np.array(all_results, dtype=object)
        np.save("comparison_results.npy", results_array)
        print(f"\n✓ Results saved to: comparison_results.npy (shape: {results_array.shape})")
        
        # Scores only array
        scores_array = np.array([row[2] for row in all_results], dtype=np.float64)
        np.save("comparison_scores.npy", scores_array)
        print(f"✓ Scores saved to: comparison_scores.npy (shape: {scores_array.shape})")
        
        # 6. Display Summary Statistics
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        
        # Group by model for summary
        model_stats = {}
        for model_name, episode, score in all_results:
            if model_name not in model_stats:
                model_stats[model_name] = []
            model_stats[model_name].append(score)
        
        # Calculate and display stats per model
        summary_data = []
        for model_name, scores in model_stats.items():
            scores_arr = np.array(scores)
            summary_data.append({
                'model': model_name,
                'mean': scores_arr.mean(),
                'std': scores_arr.std(),
                'max': scores_arr.max(),
                'count': len(scores)
            })
        
        # Sort by mean score descending
        summary_data.sort(key=lambda x: x['mean'], reverse=True)
        
        # Print formatted table
        print(f"{'Model':<30} {'Mean':>10} {'Std':>10} {'Max':>10} {'Count':>6}")
        print("-" * 70)
        for stats in summary_data:
            print(f"{stats['model']:<30} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
                  f"{stats['max']:>10.2f} {stats['count']:>6}")
        
        print("\n" + "="*60)
        print(f"Total episodes evaluated: {len(all_results)}")
        print(f"Overall mean score: {scores_array.mean():.2f}")
        print(f"Overall std: {scores_array.std():.2f}")
        print("="*60)

if __name__ == "__main__":
    main()