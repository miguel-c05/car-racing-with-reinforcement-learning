import argparse
import os
import glob
import csv
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from tqdm import tqdm

# Import local modules to ensure environment matches training exactly
from customization import make_vec_envs
import config as cfg

def setup_eval_env(num_envs=1):
    """
    Creates the environment pipeline exactly matching learning.py logic.
    Forces use_custom_rewards=False to evaluate on the 'real' metric.
    """
    # 1. Create Base Env (Custom wrappers for Grayscale/Crop, but Standard Rewards)
    env = make_vec_envs(num_envs=num_envs, use_additional_rewards=False)
    
    # 2. Frame Stack (Match Training)
    env = VecFrameStack(env, n_stack=4)
    
    # 3. Transpose (Match Training - Critical fix from learning.py)
    env = VecTransposeImage(env)
    
    return env

def evaluate_model(model_path, env, n_episodes=5, model_name="model"):
    """
    Loads a model and runs it in the environment.
    Returns a list of rewards for each episode.
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
    parser = argparse.ArgumentParser(description="Compare PPO models on the base CarRacing environment.")
    
    # Defaults based on your config.py
    parser.add_argument("--models_dir", type=str, default="./models", 
                        help="Root models directory containing version folders")
    parser.add_argument("--output_csv", type=str, default="comparison_results.csv", 
                        help="Output CSV file name")
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
    print(f"Evaluating each for {args.episodes} episodes on Base Environment (Standard Rewards)...")

    # 2. Setup Environment
    # We create ONE env and reuse it.
    env = setup_eval_env(num_envs=1)

    # 3. Evaluation Loop
    results = []

    # Prepare CSV file
    with open(args.output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "Episode", "Score"])

        pbar = tqdm(model_files, desc="Evaluating Models", position=0)
        for model_path in pbar:
            # Extract meaningful name from path
            # Path structure varies:
            # - models/re_v5/best_model/best_model.zip -> "re_v5"
            # - models/re_v6/phased/phase_1_exploration/best_model/best_model.zip -> "re_v6_phase_1_exploration"
            # - models/re_v7/phase_1_exploration/best_model/best_model.zip -> "re_v7_phase_1_exploration"
            path_parts = model_path.split(os.sep)
            
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
                if "phase_" in model_path:
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
            
            for i, r in enumerate(rewards):
                writer.writerow([model_name, i+1, r])
                results.append({"model": model_name, "reward": r})
                
                # Flush to disk immediately in case of crash
                file.flush()

    # 4. Cleanup
    env.close()
    
    print(f"\nDone! Results saved to {args.output_csv}")
    
    # 5. Quick Summary
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n--- Summary ---")
        summary = df.groupby("model")["reward"].agg(["mean", "std", "max"]).sort_values("mean", ascending=False)
        print(summary)

if __name__ == "__main__":
    main()