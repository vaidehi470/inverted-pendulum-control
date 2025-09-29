# inverted-pendulum-control
A PyTorch implementation of Deep Q-Learning to solve the CartPole-v1 environment using reinforcement learning techniques like experience replay and target networks
## Getting Started
### 1. Clone the repository (or copy the files)
git clone <your-repo-url>
cd cartpole_dqn_project
### 2. Install dependencies
We recommend using a virtual environment:
python -m venv env
env\scripts\activate
pip install -r requirements.txt
### 3. Train the model
python train.py
a. Trains the agent over 500 episodes.
b. Plots the reward over time.
c. Saves the trained model as cartpole_dqn.pth.
### 4. Evaluate
python evaluate.py
a. Loads the trained model.
b. Runs and renders the agent interacting with the CartPole environment.
c. Press Ctrl+C to manually stop the rendering loop.

### Output
a. A reward vs. episode plot will be shown.
b. Model weights saved to cartpole_dqn.pth.
c. Visual demonstration of the trained agent balancing the pole.
