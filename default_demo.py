import time


if __name__ == '__main__':
    import gymnasium as gym

    # Create the environment
    # env = gym.make('Pusher-v5', render_mode="human")
    env = gym.make("InvertedPendulum-v5", render_mode="human")#,reset_noise_scale=1)
    # env = gym.make('InvertedDoublePendulum-v5',render_mode="human", healthy_reward=10)
    max_ts = 1000

    # Reset the environment to start
    observation, info = env.reset()

    for ts in range(max_ts):  # Run for 1000 timesteps
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)  # Take a step in the environment
        hinge_deg = observation[1]*(180/3.141592653589793)


        if done:
            if truncated:
                print("Episode ended due to time limit (truncated). or success")
            # elif ts == max_ts-1 :
            #     print("Episode ended due to success (terminated).")
            else:
                print(observation[1] * (180 / 3.141592653589793))
                print("Episode ended due to failure")

            observation, info = env.reset()
            ("reset!")

        time.sleep(0.1)
    env.close()



