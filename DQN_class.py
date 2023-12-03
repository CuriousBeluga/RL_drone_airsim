import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


# Define the DQN neural network
class DQNNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQNNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.fc(x)



# Define DQN agent
class DQNAgent:
    def __init__(self, num_inputs, num_actions, gamma=0.9, epsilon=0.1, replay_buffer_capacity=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(num_inputs, num_actions).to(self.device)
        self.target_net = DQNNetwork(num_inputs, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_buffer_capacity)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.policy_net.fc[-1].out_features)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def store_experience(self, experience):
        self.replay_buffer.append(experience)

    def update_q_network(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = np.random.choice(self.replay_buffer, batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(state_tensor).gather(1, action_tensor)
        next_q_values = self.target_net(next_state_tensor).max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + (1 - done_tensor) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())