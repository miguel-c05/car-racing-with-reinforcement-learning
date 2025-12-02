import gymnasium as gym

# Try to load the environment to verify Box2D is working
try:
    env = gym.make("CarRacing-v3", render_mode="human")
    observation, info = env.reset()

    print("✅ Environment created successfully!")
    
    # Run a few steps to check physics
    for _ in range(500):
        action = env.action_space.sample()  # Random action
        env.step(action)

    env.close()
    print("✅ Physics engine test passed.")

except Exception as e:
    print(f"❌ Error loading Box2D: {e}")