
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Experience Replay Buffer
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize environment and networks
env = gym.make('CartPole-v1', render_mode=None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
replay_buffer = ReplayMemory()

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
num_episodes = 500
target_update_freq = 10

# Training Loop
episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()[0]  # unpacking observation from reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            action = policy_net(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # states = torch.FloatTensor(states)
            # actions = torch.LongTensor(actions).unsqueeze(1)
            # rewards = torch.FloatTensor(rewards).unsqueeze(1)
            # next_states = torch.FloatTensor(next_states)
            # dones = torch.FloatTensor(dones).unsqueeze(1)

            states = torch.from_numpy(np.array(states, dtype=np.float32))
            actions = torch.from_numpy(np.array(actions, dtype=np.int64)).unsqueeze(1)
            rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1)
            next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
            dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1)

            q_values = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                targets = rewards + gamma * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episode_rewards.append(total_reward)

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Plotting
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Progress of DQN on CartPole")
plt.grid(True)
plt.show()

# Save the model
torch.save(policy_net.state_dict(), "cartpole_dqn.pth")
print("Model saved as cartpole_dqn.pth")

#bool