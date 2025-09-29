import gym
import torch
import torch.nn as nn
import time
import numpy as np

# Fix for numpy compatibility with newer versions
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

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

# Load environment and model
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = QNetwork(state_dim, action_dim)
model.load_state_dict(torch.load("cartpole_dqn.pth"))
model.eval()

# Run one episode
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    time.sleep(0.02)  # Optional slow-down
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = model(state_tensor).argmax().item()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
    total_reward += reward

env.close()
print(f"Total reward in test: {total_reward}")