import gymnasium as gym
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
import cv2
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt

# --- MOCK CONFIGURATION (To make this script standalone) ---
class cfg:
    GAS_REWARD = 0.1
    WIGGLE_PENALTY = 0.5
    WIGGLE_TOLERANCE = 0.1
    OFF_ROAD_WHEEL_PENALTY = 1.0
    MAX_OFF_ROAD_STEPS = 100
    MAX_LINE_DISTANCE_REWARD = 1.0
    LINE_DISTANCE_REWARD_DROPOFF = 5.0
    TARGET_SPEED = 50.0

# --- THE WRAPPER (With Fixes applied) ---
class CustomEnvironment(gym.Wrapper):
    def __init__(self,
        env,
        gas_reward=cfg.GAS_REWARD,
        wiggle_penalty=cfg.WIGGLE_PENALTY,
        wiggle_tolerance=cfg.WIGGLE_TOLERANCE,
        off_road_wheel_penalty=cfg.OFF_ROAD_WHEEL_PENALTY,
        draw_optimal_line=True, 
        verbose=False
    ):
        super().__init__(env)
        self.gas_reward = gas_reward
        self.wiggle_penalty = wiggle_penalty
        self.wiggle_tolerance = wiggle_tolerance
        self.off_road_wheel_penalty = off_road_wheel_penalty
        self.draw_optimal_line = draw_optimal_line
        self.verbose = verbose
        
        # Internal State
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.optimal_line = None 
        
        self.info = {}
        self.info["wheels_on_road"] = 4
        self.info["consecutive_off_road"] = 0

    def reset(self, **kwargs):
        # 1. Reset Env
        obs, info = self.env.reset(**kwargs)
        
        # 2. Recalculate optimal line for the NEW track
        self.optimal_line = self.get_optimal_path(iterations=200)
        
        # 3. Reset state
        self.last_action = np.array([0.0, 0.0, 0.0])
        self.info["consecutive_off_road"] = 0
        
        # 4. Draw line on first frame
        if self.draw_optimal_line:
            obs = self.plot_optimal_line(obs)
            
        return obs, info

    def get_optimal_path(self, iterations=100):
        raw_track = self.env.unwrapped.track
        path = np.array([[p[2], p[3]] for p in raw_track])
        
        # --- CORRECTION: REAL BOX2D UNITS ---
        # The CarRacing scale is 30 pixels/unit. 
        # The visual track width constant is 40.
        # So Real Width = 40 / 30 = ~1.333 units (Half Width)
        SCALE = 30.0
        TRACK_HALF_WIDTH = 40.0 / SCALE 
        
        # Car width is roughly 1.6 units. 
        # We leave a small margin so the center of the car stays on track.
        SAFE_MARGIN = 0.4  
        max_displacement = TRACK_HALF_WIDTH - SAFE_MARGIN
        
        optimized_path = np.copy(path)
        num_points = len(path)

        for _ in range(iterations):
            for i in range(num_points):
                prev_idx = (i - 1) % num_points
                next_idx = (i + 1) % num_points
                
                midpoint = (optimized_path[prev_idx] + optimized_path[next_idx]) / 2
                optimized_path[i] = optimized_path[i] * 0.2 + midpoint * 0.8
                
                center = path[i]
                current = optimized_path[i]
                diff = current - center
                dist = np.linalg.norm(diff)
                
                if dist > max_displacement:
                    if dist > 1e-6:
                        diff = diff / dist * max_displacement
                        optimized_path[i] = center + diff
        return optimized_path

    def plot_optimal_line(self, observation, optimal_path=None, color=(0, 255, 0), thickness=1):
        """
        Draws the line on the 96x96 observation, ACCOUNTING FOR ROTATION.
        """
        if optimal_path is None:
            optimal_path = self.optimal_line
        if optimal_path is None: 
            return observation
        
        img = observation.copy()
        
        # 1. Coordinate mapping for 96x96 Agent View
        OBS_W, OBS_H = 96, 96
        
        # --- CRITICAL FIX: PIVOT POINT ---
        # The car is NOT centered at 48, 48.
        # In Gymnasium CarRacing, the car is drawn lower (roughly Y=74) to show more road ahead.
        # If we rotate around 48, we get a parallax error (pendulum effect).
        CENTER_X = 48.0
        CENTER_Y = 74.0 # Empirical pivot point correction
        
        # Scale: Pixels per Box2D unit. 
        # 6.0 was slightly too fast (gliding), 5.5 is tighter.
        SCALE = 5.5 
        
        # 2. Get Car State
        car = self.env.unwrapped.car
        car_pos = car.hull.position
        car_angle = car.hull.angle # Radians
        
        # 3. Pre-calculate rotation vectors
        # Angle 0 = North (Up) = Local Y axis
        # Forward = (-sin, cos)
        # Right   = (cos, sin)
        sin_a = np.sin(car_angle)
        cos_a = np.cos(car_angle)
        
        vec_fwd_x = -sin_a
        vec_fwd_y = cos_a
        
        vec_right_x = cos_a
        vec_right_y = sin_a

        screen_points = []
        
        for point in optimal_path:
            px, py = point
            
            # A. Vector from Car to Point (Translation)
            dx = px - car_pos.x
            dy = py - car_pos.y
            
            # Culling
            if abs(dx) > 30 or abs(dy) > 30:
                continue

            # B. Project onto Car's Local Axes (Rotation)
            local_x = (dx * vec_right_x) + (dy * vec_right_y)
            local_y = (dx * vec_fwd_x) + (dy * vec_fwd_y)

            # C. Map to Screen Pixels
            screen_x = int(CENTER_X + local_x * SCALE)
            screen_y = int(CENTER_Y - local_y * SCALE)
            
            if 0 <= screen_x < OBS_W and 0 <= screen_y < OBS_H:
                screen_points.append((screen_x, screen_y))
        
        if len(screen_points) > 1:
            points_array = np.array([screen_points], dtype=np.int32)
            # Use LINE_AA (Anti-Aliased) for smoother look
            cv2.polylines(img, points_array, isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            
        return img

    def check_early_stop(self, env):
        car = env.unwrapped.car
        wheels_on_road = 0
        for wheel in car.wheels:
            if len(wheel.tiles) > 0: wheels_on_road += 1
        return wheels_on_road

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        
        # 1. Draw Line (Visual Feedback)
        if self.draw_optimal_line:
            observation = self.plot_optimal_line(observation)
            
        # 2. Rewards (Simplified for this test script)
        reward += action[1] * self.gas_reward # Gas bias
        
        wheels_on_road = self.check_early_stop(self.env)
        if wheels_on_road == 0:
            self.info["consecutive_off_road"] += 1
        else:
            self.info["consecutive_off_road"] = 0
            
        if self.info["consecutive_off_road"] > cfg.MAX_OFF_ROAD_STEPS:
            truncated = True
        
        # Track last action
        self.last_action = action
        
        return observation, reward, done, truncated, self.info

# --- HELPER: PLOT WHOLE TRACK ---
def plot_whole_track(env, optimal_path):
    """
    Plots the entire track and the optimal line using Matplotlib.
    Blocks execution until the window is closed.
    """
    print("Plotting whole track... (Close window to continue)")
    
    # 1. Get Center Line Data
    raw_track = env.unwrapped.track
    center_line = np.array([[T[2], T[3]] for T in raw_track])
    
    # 2. Calculate Borders (Corrected for Real Box2D Scale)
    SCALE = 30.0
    TRACK_HALF_WIDTH = 40.0 / SCALE 
    
    left_boundary = []
    right_boundary = []
    num_points = len(center_line)
    
    for i in range(num_points):
        prev_idx = (i - 1) % num_points
        next_idx = (i + 1) % num_points
        
        tangent = center_line[next_idx] - center_line[prev_idx]
        norm = np.linalg.norm(tangent)
        if norm == 0: continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])
        
        current_pos = center_line[i]
        left_boundary.append(current_pos + normal * TRACK_HALF_WIDTH)
        right_boundary.append(current_pos - normal * TRACK_HALF_WIDTH)
        
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#2c3e50')
    
    # Fill track
    ax.fill(np.append(left_boundary[:,0], right_boundary[::-1,0]),
            np.append(left_boundary[:,1], right_boundary[::-1,1]),
            color='#555555', alpha=0.8, label='Track')
            
    # Plot lines
    ax.plot(center_line[:, 0], center_line[:, 1], 'w--', alpha=0.3, label="Center")
    if optimal_path is not None:
        ax.plot(optimal_path[:, 0], optimal_path[:, 1], 'r-', linewidth=2, label="Optimal")
        
    ax.set_title("Generated Track & Optimal Line")
    ax.legend()
    ax.axis('equal')
    plt.show()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Initializing CarRacing-v3...")

    # render_mode='rgb_array' is required to get the pixels for OpenCV
    env = gym.make("CarRacing-v3", render_mode="rgb_array")

    # Wrap with RecordVideo to save a clip
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True)

    # Wrap it
    env = CustomEnvironment(env, draw_optimal_line=True, verbose=True)

    obs, _ = env.reset()
    print("Environment reset. Generating optimal line...")

    # --- SHOW THE WHOLE TRACK ON STARTUP ---
    plot_whole_track(env, env.optimal_line)

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
                plot_whole_track(env, env.optimal_line) # Verify new track

    except KeyboardInterrupt:
        print("\nStoppping...")
    finally:
        env.close()
        cv2.destroyAllWindows()