#!/usr/bin/env python

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple
from itertools import count
import random

device = torch.device("cuda")


def frame_to_tensor(x):
    return torch.from_numpy((np.ascontiguousarray(x, dtype=np.float32) / 255).transpose(2, 0, 1)).view(1, 3, 210, 160)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # TODO: Add channels for last frame so we can learn about time
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(12512, 6)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.head(x.view(x.size(0), -1))
        return x


policy_net = DQN().to(device)
target_net = DQN().to(device)

optimizer = torch.optim.RMSprop(policy_net.parameters())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, eps=0.10):
    rand = random.uniform(0, 1)
    if rand > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(6)]], device=device, dtype=torch.long)


def optimize_model(buffer, batch, gamma=0.999):
    if len(buffer) < batch:
        return

    states, actions, next_states, rewards = Transition(*zip(*buffer.sample(batch)))
    # TODO: Zero out terminal states

    states = torch.cat(states)
    next_states = torch.cat(next_states)
    rewards = torch.cat(rewards)

    predicted_values = policy_net(states)[:, action].view(batch, 1)
    next_state_values = target_net(next_states).max(1)[0].view(batch, 1).detach()
    expected_values = rewards + gamma * next_state_values
    loss = F.smooth_l1_loss(predicted_values, expected_values)

    # calculate loss
    optimizer.zero_grad()
    loss.backward()
    # TODO: WTF is policy_net.parameters()? Are we clamping all gradients to [-1, 1]?
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_screen():
    return frame_to_tensor(env.render(mode='rgb_array')).to(device)


if __name__ == '__main__':
    buffer = ReplayMemory(10000)
    env = gym.make('Pong-v0')
    env.reset()
    num_episodes = 50
    for i in range(num_episodes):
        # Play an episode
        last_state = get_screen()
        current_state = get_screen()

        for t in count():
            # Play a frame
            # Variabalize 210, 160
            state = (current_state - last_state).view(1, 3, 210, 160)
            action = select_action(state)
            env.render()
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward]).view(1, 1)

            last_state = current_state
            current_state = get_screen()
            next_state = current_state - last_state
            buffer.push(state, action, next_state, reward)
            # Replay memory
            optimize_model(buffer, 128)
            if done:
                break
