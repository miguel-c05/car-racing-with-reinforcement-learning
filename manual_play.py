import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np

def manual_play(env):
    # Define key-to-action mapping for CarRacing-v3
    # Action format: [steering, gas, brake]
    # steering: -1 (left) to +1 (right)
    # gas: 0 to 1
    # brake: 0 to 1
    keys_to_action = {
        (ord('a'),): np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Steer left
        (ord('d'),): np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Steer right
        (ord('w'),): np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Gas
        (ord('s'),): np.array([0.0, 0.0, 1.0], dtype=np.float32),   # Brake
        (ord('a'), ord('w')): np.array([-1.0, 1.0, 0.0], dtype=np.float32),  # Left + Gas
        (ord('d'), ord('w')): np.array([1.0, 1.0, 0.0], dtype=np.float32),   # Right + Gas
        (ord('a'), ord('s')): np.array([-1.0, 0.0, 1.0], dtype=np.float32),  # Left + Brake
        (ord('d'), ord('s')): np.array([1.0, 0.0, 1.0], dtype=np.float32),   # Right + Brake
    }

    # No action (noop) - when no keys are pressed
    noop_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    play(env, keys_to_action=keys_to_action, noop=noop_action)
    
if __name__ == "__main__":
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env.reset()
    manual_play(env)