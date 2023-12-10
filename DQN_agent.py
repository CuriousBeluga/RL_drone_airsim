import torch
import airsim
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import os
import csv
import time
import datetime


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
    def __init__(self, num_inputs, num_actions, checkpoint_path, csv_filename, gamma=0.9, epsilon=0.1,
                 replay_buffer_capacity=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(num_inputs, num_actions).to(self.device)
        self.target_net = DQNNetwork(num_inputs, num_actions).to(self.device)
        # Check if a checkpoint file exists
        if os.path.exists(checkpoint_path):
            state_dicts = torch.load(checkpoint_path)
            self.policy_net.load_state_dict(state_dicts[0])
            self.target_net.load_state_dict(state_dicts[1])
            print(f"Successfully loaded previous model from {checkpoint_path}!")
        # else:
        #     self.policy_net = DQNNetwork(num_inputs, num_actions).to(self.device)
        #     self.target_net = DQNNetwork(num_inputs, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = deque(maxlen=replay_buffer_capacity)
        self.replay_capacity = replay_buffer_capacity
        self.gamma = gamma
        self.epsilon = epsilon

        # data collection
        self.csv_filename = csv_filename

        # Open existing CSV in append mode; else create new CSV
        mode = 'a' if os.path.exists(self.csv_filename) else 'w'
        with open(self.csv_filename, mode, newline='') as csvfile:
            fieldnames = ['Episode', 'Reward']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If it's a new file, write the header
            if mode == 'w':
                writer.writeheader()

    def select_action(self, state):
        # epsilon implementation
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.policy_net.fc[-1].out_features)
            # return 2

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

    def reset_replay_buffer(self):
        # reset the replay buffer to prevent memory issues
        self.replay_buffer = deque(maxlen=self.replay_capacity)

    def save_episode_reward(self, episode, reward, test_time):
        # Append the reward for the current episode to the CSV file
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['Episode', 'Reward','Time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'Episode': episode, 'Reward': reward,'Time': test_time})


# Set the timeout in seconds
timeout = 35

# Define a flag to indicate timeout
timeout_flag = False


# Function to set the timeout flag
def set_timeout_flag():
    global timeout_flag
    time.sleep(timeout)
    timeout_flag = True


def train_agent_obstacle_1(num_episodes,checkpoint_path,csv_file):
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()


    # Define experience tuple for replay buffer
    Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    # Set up DQN agent
    num_inputs = 3  # Assuming three state variables (x, y, z)
    num_actions = 6  # 6 actions (up, down, left, right, forward, backwards)

    epsilon_user = 0.4
    epsilon_sim = 0.7

    dqn_agent = DQNAgent(num_inputs, num_actions, checkpoint_path, csv_file, epsilon=epsilon_user)
    movement_speed = 2
    displacement = 1

    # Training loop
    # num_episodes = 100
    max_iterations = 100
    epsilon_decay = 0.995
    batch_size = 64

    for episode in range(num_episodes):
        client.reset()
        client.enableApiControl(True)
        client.takeoffAsync().join()
        # client.armDisarm(True)

        # Set the initial position of the quadcopter
        initial_pose = client.simGetVehiclePose()
        state = [initial_pose.position.x_val, initial_pose.position.y_val, initial_pose.position.z_val]

        initial_position = initial_pose.position
        # Set the target position (can use switch case to change the position later)
        target_position = airsim.Vector3r(initial_position.x_val + 35, initial_position.y_val,
                                          initial_position.z_val)

        # agent_local_target_pos = airsim.Vector3r(initial_position.x_val + 3, initial_position.y_val,
        #                                          initial_position.z_val+2)
        done = False
        total_reward = 0

        for i in range(max_iterations):
            # client.armDisarm(True)
            # client.enableApiControl(True)
            # client.confirmConnection()

            # client.enableApiControl(True)
            # agent selects action
            # if np.random.rand() < epsilon_sim:
            #     action = 2
            # else:
            #     action = dqn_agent.select_action(state)
            current_pose = client.simGetVehiclePose()
            curr_state = current_pose.position
            agent_local_target_pos = airsim.Vector3r(curr_state.x_val, curr_state.y_val,curr_state.z_val)

            action = dqn_agent.select_action(state)
            print("action calculated")
            reward = 0
            print(f'Episode {episode + 1} step: {i}')
            if action == 0:
                agent_local_target_pos.y_val += displacement  # right
            elif action == 1:
                agent_local_target_pos.y_val -= displacement  # left
            elif action == 2:
                agent_local_target_pos.x_val += displacement  # forwards
            elif action == 3:
                agent_local_target_pos.x_val -= displacement  # backwards
            elif action == 4:
                agent_local_target_pos.z_val += displacement  # up
            elif action == 5:
                agent_local_target_pos.z_val -= displacement  # down
            agent_local_target_pos.x_val += displacement  # forwards

            client.moveToPositionAsync(agent_local_target_pos.x_val, agent_local_target_pos.y_val, agent_local_target_pos.z_val, movement_speed).join()

            agent_local_target_pos.y_val = 0

            # # Set the desired command rate (commands per second)
            # command_rate = 2.0  # Adjust as needed
            #
            # # Calculate the time interval between commands
            # command_interval = 1.0 / command_rate
            # try:
            #     while True:
            #         client.armDisarm(True)
            #         client.enableApiControl(True)
            #         print("loop entered")
            #         # command agent to move with selected action
            #         client.moveToPositionAsync(curr_state.x_val, curr_state.y_val,
            #                                    curr_state.z_val, movement_speed).join()
            #         # Wait for the next command interval
            #         time.sleep(command_interval)
            #         # Get the updated position after sending the command
            #         updated_position = client.simGetVehiclePose().position
            #
            #         # Calculate the distance moved in the x-axis
            #         distance_moved = updated_position.x_val - initial_position.x_val
            #
            #         # Check if the desired distance is reached
            #         print("command loop")
            #         if distance_moved >= 0.1:
            #             break
            #
            # except KeyboardInterrupt:
            #     pass
            # finally:
            #     pass
            #     # Ensure the drone is safely landed and disarmed
            #     # client.armDisarm(False)
            #     # client.landAsync().join()
            #
            # print("command issued")
            new_pose = client.simGetVehiclePose()
            next_state = [new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val]

            # reward section based on distance tiers
            # if 35 < new_pose.position.distance_to(target_position) <= 40:
            #     reward = -35
            # elif 30 < new_pose.position.distance_to(target_position) <= 35:
            #     reward = -30
            # elif 25 < new_pose.position.distance_to(target_position) <= 30:
            #     reward = -25
            # elif 20 < new_pose.position.distance_to(target_position) <= 25:
            #     reward = -20
            # elif 15 < new_pose.position.distance_to(target_position) <= 20:
            #     reward = -15
            # if new_pose.position.distance_to(target_position) > 10:
            #     reward = -15
            # elif 5 < new_pose.position.distance_to(target_position) <= 10:
            #     reward = -10
            # elif 2 < new_pose.position.distance_to(target_position) <= 5:
            #     reward = -2

            if new_pose.position.distance_to(target_position) > 2:
                reward = -new_pose.position.distance_to(target_position)
                print(reward)
            elif new_pose.position.distance_to(target_position) <= 2:
                reward = max_iterations*35
            if new_pose.position == target_position:
                reward = 1000
                total_reward += reward
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
                print("The eagle has landed!")
                done = True
                dqn_agent.store_experience(Experience(state, action, reward, next_state, done))
                dqn_agent.update_q_network(batch_size)
                dqn_agent.update_target_network()
                # dqn_agent.save_episode_reward(episode, total_reward)
                break
            # if agent tries to exceed bounded training area, immediately terminate episode
            if new_pose.position.distance_to(target_position) > 40 or new_pose.position.z_val > 6:
                reward = -max_iterations*35
                total_reward += reward
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
                print("The eagle has failed!")
                done = True
                print(f'{new_pose.position.distance_to(target_position)} units from target')
                dqn_agent.store_experience(Experience(state, action, reward, next_state, done))
                dqn_agent.update_q_network(batch_size)
                dqn_agent.update_target_network()
                # dqn_agent.save_episode_reward(episode, total_reward)
                break

            # reward = 1.0 if new_pose.position.distance_to(target_position) < 2 else -1.0

            # Collision handling
            collision_info = client.simGetCollisionInfo()

            # Detect if collision has happened, print message statement
            if collision_info.has_collided:
                print(f"Collision occurred at time {collision_info.time_stamp}")
                print(f"Impact point: {collision_info.impact_point}")
                print(f"Collision normal: {collision_info.normal}")
                print(f"Object ID: {collision_info.object_id}")
                print(f"Object name: {collision_info.object_name}")
                done = True
                reward = -max_iterations*35
                total_reward += reward
                dqn_agent.store_experience(Experience(state, action, reward, next_state, done))
                dqn_agent.update_q_network(batch_size)
                dqn_agent.update_target_network()
                break

            # else:
            #     print("No collision has occurred.")

            # done = True if collision_info.has_collided else False

            dqn_agent.store_experience(Experience(state, action, reward, next_state, done))
            dqn_agent.update_q_network(batch_size)
            # Update the target network periodically
            if i % 10 == 0:
                dqn_agent.update_target_network()

            if i % 20 == 0:
                # reset the replay buffer every ten iterations
                dqn_agent.reset_replay_buffer()

            state = next_state
            total_reward += reward

        # Decay exploration rate
        dqn_agent.epsilon *= epsilon_decay

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # save the learned weights and biases to a file
        torch.save((dqn_agent.policy_net.state_dict(), dqn_agent.target_net.state_dict()),
                   'dqn_checkpoint_1obstacle.pth')
        # save reward for the episode
        now = datetime.datetime.now()
        dqn_agent.save_episode_reward(episode, total_reward,now)

        # if done:
        #     # if episode does not terminate then land
        #     client.landAsync().join()
        # client.enableApiControl(False)

    print("Training complete.")


# -----------------------------------------------------------------------------------------------------------
def agent_run():
    # Agent attempt to navigate to goal
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()
    client.enableApiControl(True)
    client.takeoffAsync().join()
    # Set the initial position of the quadcopter
    initial_pose = client.simGetVehiclePose()
    state = [initial_pose.position.x_val, initial_pose.position.y_val, initial_pose.position.z_val]
    initial_position = initial_pose.position
    # Set the target position (can use switch case to change the position later)
    target_position = airsim.Vector3r(initial_position.x_val + 35, initial_position.y_val, initial_position.z_val + 5)
    agent_local_target_pos = airsim.Vector3r(initial_position.x_val + 3, initial_position.y_val, initial_position.z_val)

    # Set up DQN agent
    checkpoint_path = 'dqn_checkpoint_1obstacle.pth'
    csv_file = 'dqn_1obstacle_rewards'
    num_inputs = 3  # Assuming three state variables (x, y, z)
    num_actions = 6  # 6 actions (up, down, left, right, forward, backwards)
    epsilon_user = 0
    dqn_agent = DQNAgent(num_inputs, num_actions, checkpoint_path, csv_file, epsilon=epsilon_user)

    # simulation
    movement_speed = 1
    max_steps = 100

    for e in range(max_steps):
        action = dqn_agent.select_action(state)
        if action == 0:
            agent_local_target_pos.y_val += movement_speed  # right
        elif action == 1:
            agent_local_target_pos.y_val -= movement_speed  # left
        elif action == 2:
            agent_local_target_pos.x_val += movement_speed  # forwards
        elif action == 3:
            agent_local_target_pos.x_val -= movement_speed  # backwards
        elif action == 4:
            agent_local_target_pos.z_val += movement_speed  # up
        elif action == 5:
            agent_local_target_pos.z_val -= movement_speed  # down

        # command agent to move with selected action
        client.moveToPositionAsync(agent_local_target_pos.x_val, agent_local_target_pos.y_val,
                                   agent_local_target_pos.z_val, movement_speed).join()

        new_position = client.simGetVehiclePose()
        state = [new_position.position.x_val, new_position.position.y_val, new_position.position.z_val]

        if new_position.position.distance_to(target_position) <= 2:
            print(f"The eagle has landed at step {e}!")
            break


# --------------------------------------------------------------------------------------------------------------------------
# if __name__ == "__main__":
# train agent for obstacle course 1
course_1_episodes = 100  # number of episodes in course 1
checkpoint_path = 'dqn_checkpoint_1obstacle.pth'
csv_file = 'dqn_1obstacle_rewards'
train_agent_obstacle_1(course_1_episodes, checkpoint_path, csv_file)





