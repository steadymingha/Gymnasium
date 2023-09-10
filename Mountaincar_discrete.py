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
            # nn.Linear(state_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, action_dim),
            # nn.Softmax()
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ]
        self.epsilon = 10
        self.n_actions = action_dim
        self.model = nn.Sequential(*layers).to('cuda')
        self.onpolicy_reset()
        self.train()

    def decrease_epsilon(self, decay_rate):
        self.epsilon *= decay_rate
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam
        # mean = torch.tanh(pdparam[0])
        # std = F.softplus(pdparam[1])
        # return mean, std

    def act(self, state):
        x = torch.from_numpy(state).to('cuda')
        # x = torch.FloatTensor(state[0])
        pdparam = self.forward(x)
        # mu, sigma = self.forward(x)
        pd = Categorical(logits=pdparam)
        # pd = Normal(mu, sigma)

        # if np.random.rand() < self.epsilon:
        #     # action = pd.sample()
        #     action = torch.tensor(env.action_space.sample()).to('cuda')
        # else:
        #     action = torch.argmax(pd.probs)
        action = torch.tensor(env.action_space.sample()).to('cuda')

        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)
        return action.item()
#https://stackoverflow.com/questions/67901636/taking-sample-from-categorical-distribution-pytorch


def train(pi, optimizer):
    gamma = 0.99
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t]= future_ret
    rets = torch.tensor(rets).to('cuda')
    log_probs = torch.stack(pi.log_probs).to('cuda')
    loss = - log_probs * rets
    loss = torch.sum(loss).to('cuda')
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss




if __name__ == '__main__':
    env = gym.make("MountainCar-v0", render_mode="human")
    observation, info = env.reset()

    hidden_dim = 48
    state_dim = env.observation_space.shape[0] # 2 pos, vel
    # action_dim = env.action_space.shape[0]
    action_dist_param_dim = env.action_space.n

    policy = Policy(state_dim, action_dist_param_dim, hidden_dim)
    # max_action = float(env.action_space.high[0])
    MAX_EPISODE = 100
    t_end = 400
    loss = 0

    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    epi_rewards = []

    st_bar = tqdm.tqdm(range(MAX_EPISODE))
    for epi in st_bar:
        total_reward = 0
        step = 0
        state, info = env.reset()
        # for t in range(end_t):
        # while True:
        for timestep in range(t_end):
            action = policy.act(state)
            policy.decrease_epsilon(0.999)#0.99)
            st_bar.set_postfix({'epsilon':policy.epsilon, 'loss': float(loss)})
            # state, reward, terminated, truncated, info = env.step(np.array([float(action)]))
            state, reward, terminated, truncated, info = env.step(action)
            policy.rewards.append(reward)
            env.render()
            if terminated: # or truncated:
                break

        loss = train(policy, optimizer)
        total_reward = sum(policy.rewards)
        solved = total_reward
        policy.onpolicy_reset()

        epi_rewards.append(total_reward)


    torch.save(policy.state_dict(), 'dis_ckpt.pt')


    plt.plot(range(MAX_EPISODE), epi_rewards)
    # plt.show()
    plt.savefig('dis_rewards.png')
    env.close()