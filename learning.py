import numpy as np
import stable_baselines3 as sb3
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.box2d.car_dynamics import Car
from stable_baselines3 import PPO
import cv2

class Driver():
    
    def __init__(self, vec_env, algorithm = "PPO", verbose=True):
        self.vec_env = vec_env
        self.verbose = verbose
        
        self.algorithm = algorithm.lower()
        if self.algorithm == "ppo":
            self.model = PPO(vec_env, policy="CnnPolicy", use_sde=True, verbose=self.verbose)
            





def learn(vec_env, visualize=True):
# 1. Reset everything to start
    obs = vec_env.reset()

    try:
        while True:
            # 2. Generate Random Actions for ALL environments
            # We need a list of actions, one for each environment
            # TODO: Replace with model prediction
            actions = [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]

            # 3. Step all environments simultaneously
            # 'obs' returned here is a numpy array of shape (4, 96, 96, 3)
            obs, rewards, dones, infos = vec_env.step(actions)
            
            if visualize:
                # 4. Stitch the images together to create a grid
                # Dynamically calculate grid dimensions based on number of environments
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
    
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        vec_env.close()
        cv2.destroyAllWindows()