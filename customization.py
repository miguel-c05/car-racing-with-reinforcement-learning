import config as cfg
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        
        self.info = {}
        self.info["wheels_on_road"] = 4
        self.info["consecutive_off_road"] = 0
    
    def check_early_stop(self, env):
        car : Car = env.unwrapped.car
        
        wheels_on_road = 0
        for wheel in car.wheels:
            if len(wheel.tiles) > 0:
                wheels_on_road += 1
        if self.verbose:
            if wheels_on_road == 0: print("Car off road!")
            else: print("Wheels on road:", wheels_on_road)
        
        return wheels_on_road
    
    def get_optimal_path(self, iterations=100):
        # 1. Extract the raw center line (x, y) coordinates
        # env.track is a list of (alpha, beta, x, y)
        raw_track = self.env.unwrapped.track
        path = np.array([[p[2], p[3]] for p in raw_track])

        # 2. Track Parameters
        # In CarRacing, track width is roughly constant (50 for the lane, 10 for the curb/edge). 
        # The car hull is roughly 4 units wide. We need room.
        TRACK_HALF_WIDTH = 20.0  # CarRacing track width is about 40.0 units (20.0 from center line)
        SAFE_MARGIN = 5.0        # Stay away from the edge to avoid off-road penalties
        max_displacement = TRACK_HALF_WIDTH - SAFE_MARGIN

        # Copy the path to modify it
        optimized_path = np.copy(path)
        num_points = len(path)
        
        # --- Weights for Optimization ---
        # PULL_TO_MIDPOINT: How much to smooth (move toward Prev-Next midpoint)
        PULL_TO_MIDPOINT = 0.6  # Adjust this to control smoothness (0.8 was too aggressive)
        # PULL_TO_CENTER: How much to pull back toward the original center line (stabilization)
        PULL_TO_CENTER = 1.0 - PULL_TO_MIDPOINT  # The remaining weight

        # 3. The "Rubber Band" Iteration
        for _ in range(iterations):
            # Apply a small amount of track width-based outward pull (optional, for slightly wider turns)
            # This is complex and often skipped for basic smoothing, but helps racing
            
            for i in range(num_points):
                # Get indices for previous and next points (handling the loop wrap-around)
                prev_idx = (i - 1) % num_points
                next_idx = (i + 1) % num_points

                # A. Calculate the midpoint between neighbors (The "Shortest Path" Effect)
                # The straightest line between Prev and Next passes through this midpoint
                midpoint = (optimized_path[prev_idx] + optimized_path[next_idx]) / 2

                # B. Smooth and pull back toward the original center
                
                # New point is a weighted average of the original point (stabilization) and the midpoint (smoothing)
                optimized_path[i] = optimized_path[i] * PULL_TO_CENTER + midpoint * PULL_TO_MIDPOINT

                # C. Constrain to Track Width (The "Walls")
                center = path[i]
                current = optimized_path[i]

                diff = current - center
                dist = np.linalg.norm(diff)

                if dist > max_displacement:
                    # If we moved too far off-center, clamp the point to the safe boundary
                    if dist > 1e-6: # Avoid division by zero
                        diff_unit = diff / dist
                        optimized_path[i] = center + diff_unit * max_displacement
                    else:
                        # Should not happen, but prevents crash if dist is near zero
                        optimized_path[i] = center
                        
        return optimized_path
    
    def step(self, action, last_action=None):
        observation, reward, done, truncated, info = self.env.step(action)
        
        if hasattr(info, "lap_finished"):
            self.info["lap_finished"] = info["lap_finished"]

        # ------------------
        #      GAS BIAS
        # ------------------
        gas = action[1]
        reward += gas * self.gas_reward
        
        # ------------------
        # WIGGLE PROTECTION
        # ------------------
        current_steering = action[0]  # Assuming action[0] is the steering
        if last_action is not None:
            # Additional reward for maintaining gas
            last_steering = last_action[0]
            wiggle = current_steering - last_steering
            if abs(wiggle) > self.wiggle_tolerance:
                reward -= abs(wiggle) * self.wiggle_penalty
        
        # ------------------
        # EARLY STOP LOGIC
        # ------------------
        wheels_on_road = self.check_early_stop(self.env)
        self.info["wheels_on_road"] = wheels_on_road
        if wheels_on_road == 0:
            self.info["consecutive_off_road"] += 1
        else:
            self.info["consecutive_off_road"] = 0
        
        
        if self.info["consecutive_off_road"] > cfg.MAX_OFF_ROAD_STEPS:
            truncated = True
            if self.verbose: print("Episode truncated due to excessive off-road time.")
        else: pass
        
        reward -= (4 - wheels_on_road) * self.off_road_wheel_penalty
        
        return observation, reward, done, truncated, self.info

def plot_track_analysis_accurate(env_wrapper):
    """
    Plots the track using calculated geometric borders to show TRUE physics limits.
    """
    env_wrapper.reset()
    
    # 1. Get Center Line Data
    raw_track = env_wrapper.unwrapped.track
    # raw_track items are (alpha, beta, x, y)
    center_line = np.array([[T[2], T[3]] for T in raw_track])
    
    # 2. Get Optimal Line
    # We increase the margin slightly to be safe (6.0 units from edge)
    env_wrapper.SAFE_MARGIN = 6.0 
    optimal_line = env_wrapper.get_optimal_path(iterations=200)

    # 3. Calculate Left and Right Borders Geometry
    # Track width is 40.0, so Half Width is 20.0
    TRACK_HALF_WIDTH = 20.0
    
    left_boundary = []
    right_boundary = []
    
    num_points = len(center_line)
    
    for i in range(num_points):
        # Calculate the tangent vector (direction of track)
        # We use neighbors to estimate direction
        prev_idx = (i - 1) % num_points
        next_idx = (i + 1) % num_points
        
        # Vector from prev to next
        tangent = center_line[next_idx] - center_line[prev_idx]
        
        # Normalize tangent
        norm = np.linalg.norm(tangent)
        if norm == 0: continue
        tangent = tangent / norm
        
        # Calculate Normal Vector (Perpendicular to tangent)
        # Rotate 90 degrees: (x, y) -> (-y, x)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Project borders
        current_pos = center_line[i]
        left_boundary.append(current_pos + normal * TRACK_HALF_WIDTH)
        right_boundary.append(current_pos - normal * TRACK_HALF_WIDTH)
        
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#2c3e50')
    
    # Fill the road area
    # We can fill between the boundary polygons
    # (This is a simple fill, assuming the track doesn't self-intersect wildly)
    ax.fill(np.append(left_boundary[:,0], right_boundary[::-1,0]),
            np.append(left_boundary[:,1], right_boundary[::-1,1]),
            color='#555555', alpha=0.8, label='Driveable Area')

    # Plot Borders (The "Walls")
    ax.plot(left_boundary[:, 0], left_boundary[:, 1], color='black', linewidth=1, alpha=0.5)
    ax.plot(right_boundary[:, 0], right_boundary[:, 1], color='black', linewidth=1, alpha=0.5)

    # Plot Center Line
    ax.plot(center_line[:, 0], center_line[:, 1], 
            color='white', linestyle='--', linewidth=1, alpha=0.3, label='Center')
    
    # Plot Optimal Line
    ax.plot(optimal_line[:, 0], optimal_line[:, 1], 
            color='#e74c3c', linewidth=2, label='Optimal Line')

    ax.set_title("Track Analysis: True Physics Boundaries", color='white')
    ax.legend(loc='upper right')
    ax.axis('equal') # Critical for correct proportions
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Run it
if __name__ == "__main__":
    import gymnasium as gym
    # Make sure to include your CustomEnvironment class definition here or import it
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    wrapped_env = CustomEnvironment(env)
    plot_track_analysis_accurate(wrapped_env)