import config as cfg
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing, TRACK_WIDTH
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import cv2

class CustomEnvironment(gym.Wrapper):
    def __init__(self, env, use_additional_rewards=True, offroad_penalty=False,
                 line_distance_reward=False, line_angle_reward=False,
                 drift_penalty=False, wiggle_penalty=False):
        super().__init__(env)
        self.env = env
        self.use_additional_rewards = use_additional_rewards
        self.offroad_penalty = offroad_penalty
        self.line_distance_reward = line_distance_reward
        self.line_angle_reward = line_angle_reward
        self.drift_penalty = drift_penalty
        self.wiggle_penalty = wiggle_penalty
        
        if self.use_additional_rewards:
            self.offroad_penalty = True
            self.line_distance_reward = True
            self.line_angle_reward = True
            self.drift_penalty = True
            self.wiggle_penalty = True
            
        # Cropped Observation Space (84x96)
        original_shape = self.env.observation_space.shape
        new_height = 84
        
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(new_height, original_shape[1], 1), # (84, 96, 1)
            dtype=np.uint8
        )
        
        self.consecutive_still_steps = 0
        self.consecutive_offroad_steps = 0
        self.optimal_line = None

    def get_optimal_line(self, iterations=200):
        raw_track = self.env.unwrapped.track
        path = np.array([[p[2], p[3]] for p in raw_track])
        
        # Keep the line well within track boundaries
        SAFE_MARGIN = 1.5  # More conservative margin
        max_displacement = TRACK_WIDTH - SAFE_MARGIN

        optimized_path = np.copy(path)
        num_points = len(path)

        for iteration in range(iterations):
            for i in range(num_points):
                prev_idx = (i - 1) % num_points
                next_idx = (i + 1) % num_points
                
                # Smoothing: pull towards midpoint of neighbors
                midpoint = (optimized_path[prev_idx] + optimized_path[next_idx]) / 2
                # Use more aggressive smoothing in later iterations
                blend_factor = 0.3 if iteration < iterations // 2 else 0.5
                optimized_path[i] = optimized_path[i] * (1 - blend_factor) + midpoint * blend_factor

                # Constraint: enforce maximum distance from track center
                center = path[i]
                current = optimized_path[i]
                diff = current - center
                dist = np.linalg.norm(diff)

                if dist > max_displacement:
                    diff = diff / dist * max_displacement
                    optimized_path[i] = center + diff

        return optimized_path

    def get_line_distance_and_angle_diff(self):
        """
        Returns:
        1. line_distance (float): Distance to closest point
        2. angle_diff (float): Orientation error
        3. closest_idx (int): The index of the closest point on the track
        """
        if self.optimal_line is None:
            return 0.0, 0.0, 0
            
        car = self.env.unwrapped.car
        car_pos = np.array(car.hull.position, dtype=np.float32)
        
        # 1. Define segments
        A = self.optimal_line
        B = np.roll(self.optimal_line, -1, axis=0) 
        
        # 2. Vectors
        AB = B - A              
        AP = car_pos - A        
        
        # 3. Project AP onto AB
        dot_prod = np.sum(AP * AB, axis=1)
        ab_len_sq = np.sum(AB**2, axis=1)
        
        # Avoid division by zero
        t = np.divide(dot_prod, ab_len_sq, out=np.zeros_like(dot_prod), where=ab_len_sq!=0)
        t = np.clip(t, 0.0, 1.0)
        
        # 4. Closest point C on the segment
        C = A + (AB * t[:, np.newaxis])
        
        # 5. Distances
        diff = car_pos - C
        dists = np.linalg.norm(diff, axis=1)
        
        # 6. Find Closest Index
        closest_idx = np.argmin(dists)
        line_distance = dists[closest_idx]
        
        # 7. Angle Calculation
        car_vel = car.hull.linearVelocity
        car_direction = np.array([car_vel.x, car_vel.y], dtype=np.float32)
        car_speed = np.linalg.norm(car_direction)
        
        angle_diff = 0.0
        if car_speed > 0.1:
            car_direction_norm = car_direction / car_speed
            tangent = AB[closest_idx]
            tangent_norm = tangent / np.linalg.norm(tangent)
            dot_product = np.clip(np.dot(car_direction_norm, tangent_norm), -1.0, 1.0)
            angle_diff = np.arccos(dot_product)
        
        return line_distance, angle_diff, closest_idx
    
    def get_lateral_velocity(self):
        car = self.env.unwrapped.car
        vel = car.hull.linearVelocity
        speed = np.linalg.norm([vel.x, vel.y])
        
        if speed < 0.1:
            return np.array([0.0, 0.0])
        
        vel_norm = np.array([vel.x, vel.y]) / speed
        
        # Perpendicular vector to velocity
        lateral_dir = np.array([-vel_norm[1], vel_norm[0]])
        
        lateral_velocity = np.dot(np.array([vel.x, vel.y]), lateral_dir) * lateral_dir
        return lateral_velocity
    
    def remove_observation_hud(self, observation):
        # Crop the bottom 12 pixels (HUD)
        if len(observation.shape) == 3:
            return observation[:84, :, :]
        return observation

    def process_observation(self, observation):
        # 1. Crop HUD
        obs = self.remove_observation_hud(observation)
        
        # 2. Grayscale
        # IMPORTANT: Do not use binary thresholding. 
        # We need the gray shades to see the red/white curbs.
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # 3. Add Channel Dimension for SB3 (H, W, 1)
        return np.expand_dims(gray, axis=-1)

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Process Visuals
        processed_obs = self.process_observation(observation)
        
        # --- CRITICAL FIX: Process Terminal Observation ---
        # If the episode ends (truncated/done), SB3 looks at 'terminal_observation'.
        # We must ensure this matches our observation space (84, 96, 1).
        if "terminal_observation" in info:
            info["terminal_observation"] = self.process_observation(info["terminal_observation"])
        
        # --- SIMPLIFIED REWARD SHAPING ---
        # We trust the environment's native tile-based reward system.
        
        car = self.env.unwrapped.car
        speed = np.linalg.norm(car.hull.linearVelocity)

        # 1. Wake up call: Penalty for not moving
        if speed < cfg.STILL_SPEED_THRESHOLD:
            reward -= cfg.STILL_PENALTY
            self.consecutive_still_steps += 1
        else:
            self.consecutive_still_steps = 0
        
        n_offroad_wheels = sum(1 for wheel in car.wheels if wheel.is_off_road)
        if n_offroad_wheels >= 4:
            self.consecutive_offroad_steps += 1
        else:
            self.consecutive_offroad_steps = 0
        
        
        # 2. Timeout if stuck
        if self.consecutive_still_steps > cfg.MAX_STILL_STEPS:
            truncated = True
            reward -= cfg.TRUNCATION_PENALTY # Punishment for giving up
            
        # 3. Completion Bonus
        if info.get("lap_finished"):
             reward += cfg.LAP_FINISH_BONUS

        reward = self.apply_additional_rewards(reward)
        
        return processed_obs, reward, done, truncated, info

    def apply_additional_rewards(self, reward):   
        if not self.use_additional_rewards:
            return reward

        car = self.env.unwrapped.car
        speed = np.linalg.norm(car.hull.linearVelocity)
        n_offroad_wheels = sum(1 for wheel in car.wheels if wheel.is_off_road)
        
        if self.offroad_penalty:
            if n_offroad_wheels > 0:
                reward -= n_offroad_wheels * cfg.OFFROAD_WHEEL_PENALTY
        
        if self.drift_penalty:
            lateral_velocity = self.get_lateral_velocity()
            drift_amount = np.linalg.norm(lateral_velocity)
            if drift_amount > cfg.DRIFT_THRESHOLD:
                reward -= cfg.DRIFT_PENALTY * (drift_amount / (speed + 1e-5))
        
        if self.line_distance_reward or self.line_angle_reward:
            line_distance, angle_diff, _ = self.get_line_distance_and_angle_diff()
            angle_diff_ratio = abs(angle_diff / np.pi)  # Normalize to [0, 1]
            speed_factor = min(speed / cfg.TARGET_SPEED, 1.0)
            
            if self.line_distance_reward:
                distance_reward = np.exp(-(line_distance**2) / (2 * (cfg.LINE_DISTANCE_DECAY**2)))
                distance_reward *= cfg.MAX_LINE_DISTANCE_REWARD * speed_factor
                reward += distance_reward
                
            if self.line_angle_reward:
                angle_reward = np.exp(-(angle_diff_ratio**2) / (2 * (cfg.LINE_ANGLE_DECAY**2)))
                angle_reward *= cfg.MAX_LINE_ANGLE_REWARD * speed_factor
                reward += angle_reward
            
        
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.consecutive_still_steps = 0
        self.consecutive_offroad_steps = 0
        self.optimal_line = self.get_optimal_line()
        return self.process_observation(observation), info

class CustomRepeatWrapper(gym.Wrapper):
    def __init__(self, env, repeat=cfg.ACTION_REPEAT):
        super().__init__(env)
        self.repeat = repeat

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        
        for _ in range(self.repeat):
            obs, rew, terminated, truncated_step, info = self.env.step(action)
            total_reward += rew
            if terminated or truncated_step:
                done = terminated
                truncated = truncated_step
                break
        
        return obs, total_reward, done, truncated, info

def make_custom_env(render_mode="rgb_array", use_custom_rewards=True):
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
    custom_env = CustomEnvironment(env, use_custom_rewards=use_custom_rewards)
    custom_env_repeat = CustomRepeatWrapper(custom_env)
    return Monitor(custom_env_repeat)

def make_vec_envs(num_envs=cfg.NUM_ENVS_LOW, use_custom_rewards=True):
    return make_vec_env(
        lambda: make_custom_env(use_custom_rewards=use_custom_rewards),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv
    )
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    while True:
        # Create environment and generate track
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        custom_env = CustomEnvironment(env)
        custom_env.reset()
        
        # Get track and optimal line
        raw_track = custom_env.env.unwrapped.track
        track_path = np.array([[p[2], p[3]] for p in raw_track])
        optimal_line = custom_env.optimal_line
        
        # Get actual track width from environment constants
        from gymnasium.envs.box2d.car_racing import TRACK_WIDTH, SCALE
        track_width = TRACK_WIDTH
        
        # Create track edges (left and right boundaries)
        track_left = []
        track_right = []
        
        for i in range(len(raw_track)):
            alpha, beta, x, y = raw_track[i]
            # Calculate perpendicular direction
            left_x = x - track_width * np.cos(beta)
            left_y = y - track_width * np.sin(beta)
            right_x = x + track_width * np.cos(beta)
            right_y = y + track_width * np.sin(beta)
            
            track_left.append([left_x, left_y])
            track_right.append([right_x, right_y])
        
        track_left = np.array(track_left)
        track_right = np.array(track_right)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot track boundaries
        ax.plot(track_left[:, 0], track_left[:, 1], 'k-', linewidth=1.5, alpha=0.7, label='Track Boundaries')
        ax.plot(track_right[:, 0], track_right[:, 1], 'k-', linewidth=1.5, alpha=0.7)
        
        # Fill track area
        # Create polygons for track segments
        for i in range(len(track_left)):
            i_next = (i + 1) % len(track_left)
            vertices = [
                track_left[i],
                track_right[i],
                track_right[i_next],
                track_left[i_next]
            ]
            poly = Polygon(vertices, facecolor='gray', edgecolor='none', alpha=0.3)
            ax.add_patch(poly)
        
        # Plot centerline and optimal line
        ax.plot(track_path[:, 0], track_path[:, 1], 'b--', linewidth=1.5, alpha=0.6, label='Track Center')
        ax.plot(optimal_line[:, 0], optimal_line[:, 1], 'r-', linewidth=2.5, label='Optimal Racing Line')
        
        # Mark start position
        ax.scatter(track_path[0, 0], track_path[0, 1], c='green', s=150, marker='o', 
                  label='Start', zorder=5, edgecolors='darkgreen', linewidth=2)
        
        # === DEMONSTRATION: Random point analysis using class methods ===
        # Choose a random point within the track bounds
        random_track_idx = np.random.randint(0, len(track_path))
        track_center = track_path[random_track_idx]
        # Offset from center (simulate a car position)
        random_offset = (np.random.rand(2) - 0.5) * track_width * 1.5
        random_point = track_center + random_offset
        
        # Create a random velocity vector at the point
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_velocity = np.array([np.cos(random_angle), np.sin(random_angle)]) * 20
        
        # Temporarily set car position and velocity to use the class method
        car = custom_env.env.unwrapped.car
        original_pos = car.hull.position.copy()
        original_vel = car.hull.linearVelocity.copy()
        
        # Set random position and velocity
        car.hull.position = (float(random_point[0]), float(random_point[1]))
        car.hull.linearVelocity = (float(random_velocity[0]), float(random_velocity[1]))
        
        # Use the class method to calculate everything
        line_distance, angle_diff_rad, closest_idx = custom_env.get_line_distance_and_angle_diff()
        angle_diff_deg = np.degrees(angle_diff_rad)
        
        # Calculate closest point and tangent for visualization
        A = optimal_line
        B = np.roll(optimal_line, -1, axis=0)
        AB = B - A
        AP = random_point - A
        
        dot_prod = np.sum(AP * AB, axis=1)
        ab_len_sq = np.sum(AB**2, axis=1)
        t = np.divide(dot_prod, ab_len_sq, out=np.zeros_like(dot_prod), where=ab_len_sq!=0)
        t = np.clip(t, 0.0, 1.0)
        
        C = A + (AB * t[:, np.newaxis])
        closest_point = C[closest_idx]
        
        tangent = AB[closest_idx]
        tangent_normalized = tangent / np.linalg.norm(tangent)
        
        # Restore original car state
        car.hull.position = original_pos
        car.hull.linearVelocity = original_vel
        
        # Plot the demonstration
        # Random point
        ax.scatter(random_point[0], random_point[1], c='purple', s=200, marker='*', 
                  label='Random Point', zorder=10, edgecolors='black', linewidth=2)
        
        # Distance line to closest point on optimal line
        ax.plot([random_point[0], closest_point[0]], [random_point[1], closest_point[1]], 
               'purple', linewidth=2, linestyle=':', alpha=0.8, label=f'Distance: {line_distance:.2f}')
        ax.scatter(closest_point[0], closest_point[1], c='orange', s=100, marker='x', 
                  zorder=10, linewidth=3)
        
        # Random velocity vector
        ax.arrow(random_point[0], random_point[1], random_velocity[0], random_velocity[1],
                head_width=3, head_length=2, fc='blue', ec='blue', linewidth=2, 
                alpha=0.7, label='Random Velocity', zorder=9)
        
        # Tangent vector at closest point
        tangent_scale = 25
        ax.arrow(closest_point[0], closest_point[1], 
                tangent_normalized[0] * tangent_scale, tangent_normalized[1] * tangent_scale,
                head_width=3, head_length=2, fc='cyan', ec='cyan', linewidth=2, 
                alpha=0.7, label='Line Tangent', zorder=9)
        
        # Add angle annotation
        ax.text(random_point[0] + 5, random_point[1] + 5, 
               f'Angle Diff: {angle_diff_deg:.1f}°', 
               fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'CarRacing Track (Width: {track_width*2:.1f}) with Optimal Racing Line', fontsize=14)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        env.close()
