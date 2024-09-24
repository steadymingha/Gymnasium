# from torch.distributions import Normal, Categorical
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
# import torch
# import torch.optim as optim
import gymnasium as gym
import tqdm
from matplotlib import pyplot as plt



# class Policy(torch.nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super(Policy, self).__init__()
#         layers = [
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim),
#             nn.Softmax()
#         ]
#         self.model = nn.Sequential(*layers)
#         self.onpolicy_reset()
#         self.train()
#
#     def onpolicy_reset(self):
#         self.log_probs = []
#         self.rewards = []
#
#     def forward(self, x):
#         pdparam = self.model(x)
#         return pdparam
#         # mean = torch.tanh(pdparam[0])
#         # std = F.softplus(pdparam[1])
#         # return mean, std
#
#     def act(self, state):
#         x = torch.from_numpy(state).to('cpu')
#         # x = torch.FloatTensor(state[0])
#         pdparam = self.forward(x)
#         # mu, sigma = self.forward(x)
#         pd = Categorical(logits=pdparam)
#         # pd = Normal(mu, sigma)
#
#         action = pd.sample()
#         with open('action_dis.txt', 'a') as f:
#             f.write('%d\n' % action)
#         log_prob = pd.log_prob(action)
#         self.log_probs.append(log_prob)
#         return action.item()
#
#
#
# def train(pi, optimizer):
#     gamma = 0.99
#     T = len(pi.rewards)
#     rets = np.empty(T, dtype=np.float32)
#     future_ret = 0.0
#     for t in reversed(range(T)):
#         future_ret = pi.rewards[t] + gamma * future_ret
#         rets[t]= future_ret
#     rets = torch.tensor(rets)
#     log_probs = torch.stack(pi.log_probs)
#     loss = - log_probs * rets
#     loss = torch.sum(loss)
#     optimizer.zero_grad()
#     # loss.backward(retain_graph=True)
#     loss.backward()
#     optimizer.step()
#
#     return loss




if __name__ == '__main__':
    import gymnasium as gym

    # Create the environment
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    # Reset the environment to start
    observation, info = env.reset()

    for _ in range(1000):  # Run for 1000 timesteps
        env.render()  # Render the environment to see it in action
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)  # Take a step in the environment

        if done or truncated:
            observation, info = env.reset()

    env.close()
