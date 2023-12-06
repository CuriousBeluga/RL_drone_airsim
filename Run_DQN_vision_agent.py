import airsim
from collections import namedtuple
from DQN_class import *
import cv2

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# # Enable API control for the camera
# client.simEnableApiControl(True, "front_center")

# Define experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Set up DQN agent
num_inputs = 64 * 64  # Assuming grayscale image with 64x64 resolution
num_actions = 6  # six actions (up, down, left, right, forward, backwards)
dqn_agent = DQNAgent(num_inputs, num_actions)
movement_speed = 3

# Training loop
num_episodes = 10
epsilon_decay = 0.995
batch_size = 64

initial_pose = client.simGetVehiclePose()
initial_position = initial_pose.position
# Set the target position (can use switch case to change the position later)
target_position = airsim.Vector3r(initial_position.x_val + 35, initial_position.y_val, initial_position.z_val)

for episode in range(num_episodes):
    client.reset()
    client.enableApiControl(True)
    client.takeoffAsync().join()

    # Capture and process images
    for _ in range(100):  # Example: Process images for 100 iterations
        responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene)])
        image_response = responses[0]

        if image_response is not None:

            # get numpy array
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)

            # reshape array to 4 channel image array H X W X 4
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)

            # original image is flipped vertically
            img_rgb = np.flipud(img_rgb)
            # Access image data
            image_data = image_response.image_data_uint8

            gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            # Flatten the image for input to the neural network
            flattened_image = gray_image.flatten()

            # Example: Check if obstacles are present based on pixel intensity
            obstacle_present = np.any(gray_image < 100)

            # Move towards the target position
            current_position = np.array(client.simGetVehiclePose().position)
            direction_to_target = target_position - current_position
            normalized_direction = direction_to_target / np.linalg.norm(direction_to_target)

            # Example: Map obstacle presence and target direction to actions (adjust as needed)
            if obstacle_present:
                action = 1  # Move backward
            else:
                # Move forward or turn towards the target based on direction
                forward_action = 4 if np.dot(normalized_direction, np.array([1, 0, 0])) > 0.5 else 0
                action = forward_action

            # Store the experience in the replay buffer
            state = flattened_image
            next_state = flattened_image
            reward = -1.0 if obstacle_present else 1.0
            done = False
            dqn_agent.store_experience(Experience(state, action, reward, next_state, done))

            # Update the Q-network
            dqn_agent.update_q_network(batch_size)

            # Update the target network periodically
            if _ % 10 == 0:
                dqn_agent.update_target_network()

    # Decay exploration rate
    dqn_agent.epsilon *= epsilon_decay

    print(f"Episode {episode + 1} complete.")

    client.landAsync().join()

    # # Disable API control for the camera
    # client.simEnableApiControl(False, "front_center")
