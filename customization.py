import config as cfg
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
import cv2

class CustomEnvironment(gym.Wrapper):
    def __init__(self,
        env,
        gas_reward=cfg.GAS_REWARD,
        wiggle_penalty=cfg.WIGGLE_PENALTY,
        wiggle_tolerance=cfg.WIGGLE_TOLERANCE,
        off_road_wheel_penalty=cfg.OFF_ROAD_WHEEL_PENALTY,
        draw_optimal_line=True, # New Flag
        verbose=False
    ):
        super().__init__(env)
        self.gas_reward = gas_reward
        self.wiggle_penalty = wiggle_penalty
        self.wiggle_tolerance = wiggle_tolerance
        self.off_road_wheel_penalty = off_road_wheel_penalty
        self.draw_optimal_line = draw_optimal_line
        self.verbose = verbose
        
        # State tracking
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.optimal_line = None # Will be set in reset()
        
        self.info = {}
        self.info["wheels_on_road"] = 4
        self.info["consecutive_off_road"] = 0
    
    def reset(self, **kwargs):
        # 1. Reset the simulation
        obs, info = self.env.reset(**kwargs)
        
        # 2. CRITICAL: Recalculate optimal line for the NEW track
        self.optimal_line = self.get_optimal_path(iterations=200)
        
        # 3. Reset internal state
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.info["consecutive_off_road"] = 0
        
        # 4. Optional: Draw the line on the very first frame
        if self.draw_optimal_line:
            obs = self.render_optimal_line(obs)
            
        return obs, info
    
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
    
    def render_optimal_line(self, observation):
        """ Draws the optimal line on the frame so the CNN can see it. """
        if self.optimal_line is None: return observation
        
        car = self.env.unwrapped.car
        img = observation.copy() # Don't mutate original if possible
        
        # Constants for coordinate transformation
        WINDOW_W, WINDOW_H = 600, 400
        SCALE = 6.0
        
        # Optimize: Only transform points near the car to save FPS
        # (For simplicity here, we transform all, but in production you'd clip)
        
        screen_points = []
        for point in self.optimal_line:
            x, y = point
            screen_x = int(WINDOW_W / 2 + (x - car.hull.position.x) * SCALE)
            screen_y = int(WINDOW_H / 2 + (y - car.hull.position.y) * SCALE)
            
            # Simple bounds check to avoid drawing off-canvas
            if -100 < screen_x < 700 and -100 < screen_y < 500:
                screen_points.append((screen_x, screen_y))
        
        if len(screen_points) > 1:
            # Draw green line, thickness 1
            cv2.polylines(img, [np.array(screen_points)], isClosed=True, color=(0, 255, 0), thickness=1)
            
        return img
    
    def plot_optimal_line(self, observation, optimal_path=None, color=(0, 255, 0), thickness=1):
        """
        Plots the optimal racing line on the 96x96 agent observation.
        """
        # 1. Fallback: Use the cached line if none is provided
        if optimal_path is None:
            optimal_path = self.optimal_line
        
        # If we still don't have a path (e.g., before reset), return original image
        if optimal_path is None: 
            return observation
        
        # 2. Setup Coordinate System for 96x96 Observation
        # The agent sees a 96x96 crop centered roughly on the car.
        img = observation.copy()
        
        # CarRacing-v2 constants for the 96x96 view
        # The car is always centered in the middle of the 96x96 image
        OBS_W, OBS_H = 96, 96
        CENTER_X, CENTER_Y = 48, 48 
        
        # SCALE: This is "Pixels per Box2D World Unit"
        # In the 96x96 observation, the zoom level is approx 6.0 (Standard Box2D scale)
        SCALE = 6.0 
        
        # 3. Get Car Position (The "Camera Center")
        # We need the car's position to know where to shift the world points
        car_pos = self.env.unwrapped.car.hull.position
        
        # 4. Transform World Coordinates to Screen Coordinates
        screen_points = []
        
        # Optimization: Only process points roughly near the car 
        # (World is 1000x1000, we only see a 16x16 patch)
        # We assume coordinates are within ~15 units to be visible
        
        for point in optimal_path:
            x, y = point
            
            # Calculate distance from car to point
            diff_x = x - car_pos.x
            diff_y = y - car_pos.y
            
            # Quick check: Is the point even close to the camera? (Optimization)
            if abs(diff_x) > 20 or abs(diff_y) > 20:
                continue

            # Transform:
            # 1. Shift world so car is at (0,0) -> (diff_x, diff_y)
            # 2. Scale up to pixels -> (diff_x * SCALE)
            # 3. Shift to screen center -> (+ CENTER_X)
            screen_x = int(CENTER_X + diff_x * SCALE)
            screen_y = int(CENTER_Y - diff_y * SCALE) # SUBTRACT Y because pixels go down, world goes up
            
            # Add to list if it fits on screen
            if 0 <= screen_x < OBS_W and 0 <= screen_y < OBS_H:
                screen_points.append((screen_x, screen_y))
        
        # 5. Draw
        if len(screen_points) > 1:
            # We use polylines to draw connected segments
            # Note: cv2 expects a list of arrays of points
            points_array = np.array([screen_points], dtype=np.int32)
            cv2.polylines(img, points_array, isClosed=False, color=color, thickness=thickness)
            
        return img
    
    def get_line_distance(self):
        car_pos = np.array(self.env.unwrapped.car.hull.position, dtype=np.float32)
        
        # 1. Define the segments
        # Points A are the current path points
        # Points B are the next points (shifted by 1)
        # We assume self.optimal_path is shape (N, 2)
        A = self.optimal_line
        B = np.roll(self.optimal_line, -1, axis=0) # Shift entire array by -1 to get next points
        
        # 2. Vectors
        AB = B - A              # Vector of the road segment
        AP = car_pos - A        # Vector from segment start to car
        
        # 3. Project AP onto AB to find "t"
        # Dot product of (N, 2) arrays along axis 1
        dot_prod = np.sum(AP * AB, axis=1)
        
        # Squared length of AB (avoid sqrt for speed)
        ab_len_sq = np.sum(AB**2, axis=1)
        
        # Avoid division by zero (just in case of duplicate points)
        # If segment length is 0, t doesn't matter (closest is just A)
        t = np.divide(dot_prod, ab_len_sq, out=np.zeros_like(dot_prod), where=ab_len_sq!=0)
        
        # 4. Clamp t to segment bounds [0, 1]
        t = np.clip(t, 0.0, 1.0)
        
        # 5. Calculate the closest point C on the segment
        # We need to reshape t from (N,) to (N, 1) for broadcasting
        C = A + (AB * t[:, np.newaxis])
        
        # 6. Euclidean distance from Car P to Closest Point C
        diff = car_pos - C
        dists = np.linalg.norm(diff, axis=1)
        
        # 7. The absolute minimum distance across all segments
        return np.min(dists)
    
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
    
    def step(self, action):
        # Note: 'last_action' removed from arguments to match Gym API
        
        observation, reward, done, truncated, info = self.env.step(action)
        
        if hasattr(info, "lap_finished"):
            self.info["lap_finished"] = info["lap_finished"]

        # --- OPTIONAL: VISUAL GUIDANCE ---
        if self.draw_optimal_line:
            observation = self.render_optimal_line(observation)

        # ------------------
        #      GAS BIAS
        # ------------------
        gas = action[1]
        reward += gas * self.gas_reward
        
        # ------------------
        # WIGGLE PROTECTION
        # ------------------
        current_steering = action[0]
        last_steering = self.last_action[0] # Use internal state
        wiggle = current_steering - last_steering
        
        if abs(wiggle) > self.wiggle_tolerance:
            reward -= abs(wiggle) * self.wiggle_penalty
        
        # Update state for next frame
        self.last_action = action 
        
        # -----------------------------
        # OPTIMAL LINE CLOSENESS REWARD
        # -----------------------------
        line_distance = self.get_line_distance()
        
        # Use numpy for generic vector calculation
        car_vel = self.env.unwrapped.car.hull.linearVelocity
        car_speed = np.linalg.norm([car_vel.x, car_vel.y])
        
        reward += self.line_distance_reward_function(line_distance, car_speed)
        
        # ----------------
        # EARLY STOP LOGIC
        # ----------------
        wheels_on_road = self.check_early_stop(self.env)
        self.info["wheels_on_road"] = wheels_on_road
        
        if wheels_on_road == 0:
            self.info["consecutive_off_road"] += 1
        else:
            self.info["consecutive_off_road"] = 0
        
        if self.info["consecutive_off_road"] > cfg.MAX_OFF_ROAD_STEPS:
            truncated = True
            # Optional: Add a large penalty for giving up
            reward -= 5.0
        
        reward -= (4 - wheels_on_road) * self.off_road_wheel_penalty
        
        return observation, reward, done, truncated, self.info

if __name__ == "__main__":
    print("Initializing CarRacing-v3...")
    
    # render_mode='rgb_array' is required to get the pixels for OpenCV
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    # Wrap it
    env = CustomEnvironment(env, draw_optimal_line=True, verbose=True)
    
    obs, _ = env.reset()
    print("Environment reset. Generating optimal line...")
    
    try:
        while True:
            # Action: [Steering, Gas, Brake]
            # Simple policy: Drive forward slowly
            action = np.array([0.0, 0.2, 0.0])
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- VISUALIZATION ---
            # Gymnasium returns RGB, OpenCV needs BGR
            display_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            
            # Upscale the 96x96 image to 500x500 so we can actually see it
            display_img = cv2.resize(display_img, (500, 500), interpolation=cv2.INTER_NEAREST)
            
            cv2.imshow("Agent View (Green Line = Optimal)", display_img)
            
            # Check for Quit (Press 'q')
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            
            # Reset if episode ends
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStoppping...")
    finally:
        env.close()
        cv2.destroyAllWindows()