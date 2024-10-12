import time
from matplotlib import pyplot as plt
import numpy as np

# Xdot = aX+bU
# (X_k-X_k-1)/dt = a(X_k-1)+bU
# X_k-X_k-1 = a*dt*(X_k-1) + b*dt*U
# X_k = (a*dt+1)*X_k-1 + b*dt*U

if __name__ == '__main__':
    import gymnasium as gym

    # Create the environment
    # env = gym.make('Pusher-v5', render_mode="human")
    env = gym.make("InvertedPendulum-v5")#, render_mode="human")#,reset_noise_scale=1)
    # env = gym.make('InvertedDoublePendulum-v5',render_mode="human", healthy_reward=10)
    max_ts = 1000
    control_var = np.zeros((4,4))
    responses = []
    action_max, action_min = 3, -3
    response_max, response_min = 20, -20



    # Reset the environment to start
    observation, info = env.reset()
    action = env.action_space.sample()  # Random action

    for ts in range(max_ts):  # Run for 1000 timesteps
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)  # Take a step in the environment
        hinge_deg = observation[1]*(180/3.141592653589793)

        target_deg = 0
        error_feedback = target_deg - hinge_deg
        p_gain = 5
        i_gain = 0.1
        d_gain = 0.1

        control_input = error_feedback * p_gain

        # -3+(((input_1+20)/40)*6)
        # control_normed = ((input_1+20)/40)
        # action_normed = -3 + (control_normed*6)
        control_normed = (control_input + response_max) / (response_max - response_min)
        action_normed = action_min + (control_normed * (action_max - action_min))
        action = action_normed

        responses.append(error_feedback)

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
        # env.render()  # Render the environment to see it in action
        # if abs(hinge_deg) > 20:
        #     print(observation[1] * (180 / 3.141592653589793))
        #     test = 0
    # if abs(observation[1]) <0.2:
    #     time.sleep(0.1)
    env.close()

    plt.plot(responses)
    plt.show()



