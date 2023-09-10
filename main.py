from torch.distributions import Normal, Categorical
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import tqdm
from matplotlib import pyplot as plt



class Policy(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        mean = torch.tanh(pdparam[0])
        std = F.softplus(pdparam[1])
        return mean, std

    def act(self, state):
        x = torch.from_numpy(state)#.to('cuda')
        # x = torch.FloatTensor(state[0])
        # pdparam = self.forward(x)
        mu, sigma = self.forward(x)
        # pd = Categorical(logits=pdparam)
        pd = Normal(mu, sigma)

        action = pd.sample()
        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()



def train(pi, optimizer):
    gamma = 0.99
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t]= future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets
    loss = torch.sum(loss)
    optimizer.zero_grad()
    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()

    return loss

#경사항, 최대화를 위해 음의 부호로 함
# 역전파, 경사를 계산
#경사 상승, 가중치를 업데이트



if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0", render_mode="human")
    observation, info = env.reset()

    hidden_dim = 32
    state_dim = env.observation_space.shape[0] # 2 pos, vel
    # action_dim = env.action_space.shape[0]
    action_dist_param_dim = 2 #env.action_space.shape[0]

    policy = Policy(state_dim, action_dist_param_dim, hidden_dim)
    max_action = float(env.action_space.high[0])
    MAX_EPISODE = 100
    end_t = 999

    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    epi_rewards = []

    for epi in tqdm.tqdm(range(MAX_EPISODE)):
        total_reward = 0
        step = 0
        state = env.reset()[0]

        # for t in range(end_t):
        while True:
            action = policy.act(state)
            state, reward, terminated, truncated, info = env.step(np.array([float(action)]))
            policy.rewards.append(reward)
            # env.render()
            if terminated or truncated:
                break

        loss = train(policy, optimizer)
        total_reward = sum(policy.rewards)
        solved = total_reward
        policy.onpolicy_reset()

        epi_rewards.append(total_reward)

    torch.save(policy.state_dict(), 'ckpt.pt')


    plt.plot(range(MAX_EPISODE), epi_rewards)
    # plt.show()
    plt.savefig('rewards.png')
    env.close()