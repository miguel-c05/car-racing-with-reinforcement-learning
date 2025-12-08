import config as cfg
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.box2d.car_dynamics import Car
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import cv2

class CustomEnvironment(gym.Wrapper):
    def __init__(self,
        env,
        gas_reward=cfg.GAS_REWARD,
        wiggle_penalty=cfg.WIGGLE_PENALTY,
        wiggle_tolerance=cfg.WIGGLE_TOLERANCE,
        off_road_wheel_penalty=cfg.OFF_ROAD_WHEEL_PENALTY,
        verbose=False
    ):
        super().__init__(env)
        self.gas_reward = gas_reward
        self.wiggle_penalty = wiggle_penalty
        self.wiggle_tolerance = wiggle_tolerance
        self.off_road_wheel_penalty = off_road_wheel_penalty
        self.verbose = verbose
        
        # State tracking
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.optimal_line = None # Will be set in reset()
        
        self.info = {}
        self.info["wheels_on_road"] = 4
        self.info["consecutive_off_road"] = 0

        # --- CORREÇÃO IMPORTANTE ---
        # Como vamos cortar a imagem, precisamos atualizar o observation_space
        # O original é (96, 96, 3). Vamos remover 12 pixels de baixo, ficando (84, 96, 3).
        original_shape = self.env.observation_space.shape
        new_height = 84
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(new_height, original_shape[1], original_shape[2]), 
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        # 1. Reset the simulation
        obs, info = self.env.reset(**kwargs)
        
        # 2. CRITICAL: Recalculate optimal line for the NEW track
        self.optimal_line = self.get_optimal_path(iterations=200)
        
        # 3. Reset internal state
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.info["consecutive_off_road"] = 0
        
        # 4. Aplica o corte na observação inicial
        obs = self.remove_observation_hud(obs)
            
        return obs, info
    
    def check_early_stop(self, env):
        car : Car = env.unwrapped.car
        
        wheels_on_road = 0
        for wheel in car.wheels:
            if len(wheel.tiles) > 0:
                wheels_on_road += 1
        
        return wheels_on_road
    
    def get_optimal_path(self, iterations=100):
        # 1. Extract the raw center line (x, y) coordinates
        # env.track is a list of (alpha, beta, x, y)
        raw_track = self.env.unwrapped.track
        path = np.array([[p[2], p[3]] for p in raw_track])

        # 2. Track Parameters
        # In CarRacing, track width is roughly constant. 
        # We define a "Safe Width" slightly smaller than the real width so the agent doesn't clip the grass.
        TRACK_WIDTH = 40.0 
        SAFE_MARGIN = 6.0  # Stay away from the absolute edge
        max_displacement = (TRACK_WIDTH / 2) - SAFE_MARGIN

        # Copy the path to modify it
        optimized_path = np.copy(path)
        num_points = len(path)

        # 3. The "Rubber Band" Iteration
        for _ in range(iterations):
            for i in range(num_points):
                # Get indices for previous and next points (handling the loop wrap-around)
                prev_idx = (i - 1) % num_points
                next_idx = (i + 1) % num_points

                # A. Calculate the midpoint between neighbors
                # The straightest line between Prev and Next passes through this midpoint
                midpoint = (optimized_path[prev_idx] + optimized_path[next_idx]) / 2

                # B. Move the current point towards that midpoint (Smoothing)
                # This creates the "Shortest Path" effect
                # 0.5 means we move halfway there. 
                optimized_path[i] = optimized_path[i] * 0.2 + midpoint * 0.8

                # C. Constrain to Track Width (The "Walls")
                # We cannot let the point leave the road.
                # Calculate distance from the original center line (path[i])
                center = path[i]
                current = optimized_path[i]

                diff = current - center
                dist = np.linalg.norm(diff)

                if dist > max_displacement:
                    # If we pulled too tight and hit the wall, clamp it back to the edge
                    diff = diff / dist * max_displacement
                    optimized_path[i] = center + diff

        return optimized_path
    
    def get_line_distance_and_angle_diff(self):
        """
        Returns both the minimum distance to the optimal line and the angle difference
        between car direction and the tangent of the closest segment.
        """
        car = self.env.unwrapped.car
        car_pos = np.array(car.hull.position, dtype=np.float32)
        
        # 1. Define the segments
        A = self.optimal_line
        B = np.roll(self.optimal_line, -1, axis=0) # Shift entire array by -1 to get next points
        
        # 2. Vectors
        AB = B - A              # Vector of the road segment (tangent vectors)
        AP = car_pos - A        # Vector from segment start to car
        
        # 3. Project AP onto AB to find "t"
        dot_prod = np.sum(AP * AB, axis=1)
        ab_len_sq = np.sum(AB**2, axis=1)
        
        # Avoid division by zero
        t = np.divide(dot_prod, ab_len_sq, out=np.zeros_like(dot_prod), where=ab_len_sq!=0)
        
        # 4. Clamp t to segment bounds [0, 1]
        t = np.clip(t, 0.0, 1.0)
        
        # 5. Calculate the closest point C on the segment
        C = A + (AB * t[:, np.newaxis])
        
        # 6. Euclidean distance from Car P to Closest Point C
        diff = car_pos - C
        dists = np.linalg.norm(diff, axis=1)
        
        # 7. Find the index of the closest segment
        closest_idx = np.argmin(dists)
        line_distance = dists[closest_idx]
        
        # 8. Calculate angle difference between car velocity and tangent
        car_vel = car.hull.linearVelocity
        car_direction = np.array([car_vel.x, car_vel.y], dtype=np.float32)
        car_speed = np.linalg.norm(car_direction)
        
        if car_speed < 0.1:
            return line_distance, 0.0
        
        car_direction_norm = car_direction / car_speed
        tangent = AB[closest_idx]
        tangent_norm = tangent / np.linalg.norm(tangent)
        
        dot_product = np.clip(np.dot(car_direction_norm, tangent_norm), -1.0, 1.0)
        angle_diff = np.arccos(dot_product)
        
        return line_distance, angle_diff
    
    def line_distance_reward_function(self,
        line_distance : float,
        car_speed : float = 0.0,
        max_reward = cfg.MAX_LINE_DISTANCE_REWARD,
        dropoff =cfg.LINE_DISTANCE_REWARD_DROPOFF,
        target_speed = cfg.TARGET_SPEED
    ):
        speed_factor = min(car_speed / target_speed, 1.0)
        f = max_reward * np.exp(- (line_distance ** 2) / (2 * (dropoff ** 2))) * speed_factor
        
        return f
    
    def line_angle_diff_reward_function(self,
        angle_diff : float,
        max_reward = cfg.MAX_ANGLE_DIFF_REWARD,
        dropoff = cfg.ANGLE_DIFF_REWARD_DROPOFF,
    ):
        f = max_reward * np.exp(- (angle_diff ** 2) / (2 * (dropoff ** 2)))
        return f
    
    def remove_observation_hud(self, observation):
        """
        Removes the dashboard (HUD) from the bottom of the observation.
        The CarRacing-v3 HUD is 12 pixels high.
        Original: (96, 96, 3) -> Cropped: (84, 96, 3)
        """
        # Se for None (pode acontecer em resets quebrados), retorna None
        if observation is None:
            return None
            
        # Handle batched input (N_Envs, Height, Width, Channels)
        if len(observation.shape) == 4:
            return observation[:, :84, :, :]
        # Handle single input (Height, Width, Channels)
        elif len(observation.shape) == 3:
            return observation[:84, :, :]
        return observation
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        if hasattr(info, "lap_finished"):
            self.info["lap_finished"] = info["lap_finished"]

        # ----------------------------------------
        #                GAS BIAS
        # ----------------------------------------
        gas = action[1]
        reward += gas * self.gas_reward
        
        # ----------------------------------------
        #            WIGGLE PROTECTION
        # ----------------------------------------
        current_steering = action[0]
        last_steering = self.last_action[0]
        wiggle = current_steering - last_steering
        
        if abs(wiggle) > self.wiggle_tolerance:
            reward -= abs(wiggle) * self.wiggle_penalty
        
        self.last_action = action 
        
        line_distance, angle_diff = self.get_line_distance_and_angle_diff()
        car_vel = self.env.unwrapped.car.hull.linearVelocity
        car_speed = np.linalg.norm([car_vel.x, car_vel.y])
        
        # ======== PID REWARD COMPONENTS =========
        reward += self.line_distance_reward_function(line_distance, car_speed)
        reward += self.line_angle_diff_reward_function(angle_diff)
        
        # ----------------------------------------
        #            EARLY STOP LOGIC
        # ----------------------------------------
        wheels_on_road = self.check_early_stop(self.env)
        self.info["wheels_on_road"] = wheels_on_road
        
        if wheels_on_road == 0:
            self.info["consecutive_off_road"] += 1
        else:
            self.info["consecutive_off_road"] = 0
        
        if self.info["consecutive_off_road"] > cfg.MAX_OFF_ROAD_STEPS:
            truncated = True
            reward -= 5.0
        
        # ----------------------------------------
        #         OFF-ROAD WHEEL PENALTY
        # ----------------------------------------
        reward -= (4 - wheels_on_road) * self.off_road_wheel_penalty
        
        # Aplica o corte
        clean_observation = self.remove_observation_hud(observation)
        
        return clean_observation, reward, done, truncated, self.info

if __name__ == "__main__":
    print("Initializing CarRacing-v3...")
    
    # render_mode='rgb_array' ainda é necessário para o backend funcionar, 
    # mas não usaremos o 'render()' para visualização direta
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Wrap it
    env = CustomEnvironment(env)
    
    obs, _ = env.reset()
    print(f"Environment reset. Observation shape: {obs.shape}") # Deve ser (84, 96, 3)
    
    try:
        while True:
            # Action: [Steering, Gas, Brake]
            action = np.array([0.0, 0.2, 0.0])
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- VISUALIZAÇÃO DO AGENTE (CORTADA) ---
            # 'obs' é o que o agente vê. O Gym retorna RGB, mas o OpenCV usa BGR.
            # Precisamos converter para as cores ficarem certas na janela.
            frame_agent = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            # Upscale the image for better visualization
            frame_agent_upscaled = cv2.resize(frame_agent, (480, 420), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("Agent View (Cropped)", frame_agent_upscaled)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Reset if episode ends
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()
        cv2.destroyAllWindows()
        
def make_custom_env(render_mode="rgb_array"):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    custom_env = CustomEnvironment(env)
    custom_env = Monitor(custom_env)
    custom_env.reset()
    return custom_env

def make_vec_envs(num_envs=4):
    vec_env = make_vec_env(
        make_custom_env,
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )
    vec_env.reset()
    return vec_env