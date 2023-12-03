import airsim
import numpy as np

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Set up Q-learning parameters
num_actions = 4  # Assuming four discrete actions (up, down, left, right)
num_states = 3  # Assuming three state variables (x, y, z)

# Initialize Q-table
Q_table = {}

# Define state and action space
state_space = np.linspace(-50, 50, num_states)
action_space = [0, 1, 2, 3]  # Assuming 0: up, 1: down, 2: left, 3: right

# Set the initial position of the quadcopter
initial_pose = client.simGetVehiclePose()
initial_position = initial_pose.position
# Set the target position (can use switch case to change the position)
target_position = airsim.Vector3r(initial_position.x_val, initial_position.y_val-35, initial_position.z_val)


# Define movement speed
movement_speed = 3

# Training loop
num_episodes = 10
epsilon = 0.1  # Exploration-exploitation trade-off
for episode in range(num_episodes):
    # Reset the environment
    client.reset()
    client.enableApiControl(True)
    client.takeoffAsync().join()

    # Get the initial state
    initial_pose = client.simGetVehiclePose()
    current_state = tuple([initial_pose.position.x_val, initial_pose.position.y_val, initial_pose.position.z_val])

    done = False
    total_reward = 0

    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(Q_table.get(current_state, np.zeros(num_actions)))

        # Execute action and get next state
        if action == 0:
            target_position.y_val += movement_speed
        elif action == 1:
            target_position.y_val -= movement_speed
        elif action == 2:
            target_position.x_val -= movement_speed
        elif action == 3:
            target_position.x_val += movement_speed

        # Move to the new position
        client.moveToPositionAsync(target_position.x_val, target_position.y_val, target_position.z_val, movement_speed).join()

        # Get the new state
        new_pose = client.simGetVehiclePose()
        new_state = tuple([new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val])

        # Calculate reward (e.g., penalize for being too close to obstacles)
        reward = 1.0 if new_pose.position.distance_to(target_position) < 2 else -1.0

        # Update Q-table using the Bellman equation
        current_q_values = Q_table.get(current_state, np.zeros(num_actions))
        next_q_values = Q_table.get(new_state, np.zeros(num_actions))
        current_q_values[action] += 0.1 * (reward + 0.9 * np.max(next_q_values) - current_q_values[action])
        Q_table[current_state] = current_q_values

        # Update state and total reward
        current_state = new_state
        total_reward += reward

        # Check if the episode is done
        if new_pose.position.distance_to(target_position) < 1:
            done = True

    # Print the total reward for the episode
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Land the quadcopter
    client.landAsync().join()

print("Training complete.")