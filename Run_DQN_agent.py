import airsim
from collections import namedtuple
from DQN_class import *
# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()





# Define experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


# Set up DQN agent
num_inputs = 3  # Assuming three state variables (x, y, z)
num_actions = 4  # Assuming four discrete actions (up, down, left, right)
dqn_agent = DQNAgent(num_inputs, num_actions)
movement_speed = 3

# Training loop
num_episodes = 10
epsilon_decay = 0.995
batch_size = 64

for episode in range(num_episodes):
    client.reset()
    client.enableApiControl(True)
    client.takeoffAsync().join()

    # Set the initial position of the quadcopter
    initial_pose = client.simGetVehiclePose()
    state = [initial_pose.position.x_val, initial_pose.position.y_val, initial_pose.position.z_val]

    initial_position = initial_pose.position
    # Set the target position (can use switch case to change the position later)
    target_position = airsim.Vector3r(initial_position.x_val, initial_position.y_val - 35, initial_position.z_val)

    done = False
    total_reward = 0

    while not done:
        action = dqn_agent.select_action(state)

        if action == 0:
            target_position.y_val += movement_speed
        elif action == 1:
            target_position.y_val -= movement_speed
        elif action == 2:
            target_position.x_val -= movement_speed
        elif action == 3:
            target_position.x_val += movement_speed

        client.moveToPositionAsync(target_position.x_val, target_position.y_val, target_position.z_val, movement_speed).join()

        new_pose = client.simGetVehiclePose()       # Please investigate the fields for this client object
        next_state = [new_pose.position.x_val, new_pose.position.y_val, new_pose.position.z_val]

        reward = 1.0 if new_pose.position.distance_to(target_position) < 2 else -1.0
        done = True if new_pose.collision.has_collided else False

        dqn_agent.store_experience(Experience(state, action, reward, next_state, done))
        dqn_agent.update_q_network(batch_size)
        dqn_agent.update_target_network()

        state = next_state
        total_reward += reward

    # Decay exploration rate
    dqn_agent.epsilon *= epsilon_decay

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    client.landAsync().join()

print("Training complete.")