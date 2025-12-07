from customization import CustomEnvironment
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import cv2
import numpy as np
import time

def make_custom_env(seed=0, render_mode="rgb_array"):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    custom_env = CustomEnvironment(env)
    custom_env = Monitor(custom_env)
    
    custom_env.reset(seed=seed)
    
    return custom_env

if __name__ == "__main__":

    vec_env = make_vec_env(
        make_custom_env,
        n_envs=4,
        vec_env_cls= SubprocVecEnv
    )

    print(f"Environment Vectorized! Managing {vec_env.num_envs} cars at once.")

    # 1. Reset everything to start
    obs = vec_env.reset()

    try:
        while True:
            # 2. Generate Random Actions for ALL environments
            # We need a list of actions, one for each environment
            actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]

            # 3. Step all environments simultaneously
            # 'obs' returned here is a numpy array of shape (4, 96, 96, 3)
            start_time_step = time.perf_counter()
            obs, rewards, dones, infos = vec_env.step(actions)
            end_time_step = time.perf_counter()
            step_time = (end_time_step - start_time_step) * 1000  # Convert to milliseconds
            print(f"Step time (Phase 3): {step_time:.2f} ms")

            # 4. Stitch the images together to create a grid
            # Dynamically calculate grid dimensions based on number of environments
            
            start_time_show = time.perf_counter()
            
            n_envs = vec_env.num_envs
            
            # Calculate grid dimensions: prefer square-ish layouts
            # e.g., 4 -> (2,2), 8 -> (3,3), 9 -> (3,3), 12 -> (3,4)
            grid_cols = int(np.ceil(np.sqrt(n_envs)))
            grid_rows = int(np.ceil(n_envs / grid_cols))
            
            # Build rows
            rows = []
            for r in range(grid_rows):
                row_images = []
                for c in range(grid_cols):
                    idx = r * grid_cols + c
                    if idx < n_envs:
                        row_images.append(obs[idx])
                    else:
                        # Pad with black image if we don't have enough environments
                        black_img = np.zeros_like(obs[0])
                        row_images.append(black_img)
                
                rows.append(np.hstack(row_images))
            
            # Stack all rows vertically
            grid_image = np.vstack(rows)

            # 5. Convert RGB (Gym) to BGR (OpenCV)
            grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)

            # 6. Resize for easier viewing (Optional: Make it bigger)
            # Scale to 800 pixels on the longer dimension
            grid_image_big = cv2.resize(grid_image_bgr, (800, 800), interpolation=cv2.INTER_NEAREST)

            # 7. Display
            cv2.imshow(f"Vectorized Environment ({n_envs}x) - {grid_rows}x{grid_cols} grid", grid_image_big)
            
            end_time_show = time.perf_counter()
            visualization_time = (end_time_show - start_time_show) * 1000  # Convert to milliseconds
            print(f"Visualization time (Phase 4): {visualization_time:.2f} ms")

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        vec_env.close()
        cv2.destroyAllWindows()