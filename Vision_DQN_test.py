import torch
from DQN_class import *



# Assuming num_inputs and num_actions are defined
num_inputs = 64*64  # Replace with the actual number of features in your state space
num_actions = 6  # Replace with the actual number of possible actions

# Create an instance of the DQNNetwork
network = DQNNetwork(num_inputs, num_actions)



# Create a sample input tensor
sample_input = torch.randn(1, num_inputs)  # Adjust the batch size if necessary

# Pass the sample input through the network
output = network(sample_input)

# Print the output shape
print("Output Shape:", output.shape)


# test the agent class
dqn_agent = DQNAgent(num_inputs, num_actions)
# Test the select_action method
test_state = np.random.random((num_inputs,))
action = dqn_agent.select_action(test_state)
print(f"Selected action: {action}")

# Test the store_experience method
experience = (test_state, action, 1.0, np.random.random((num_inputs,)), False)
dqn_agent.store_experience(experience)
print("Experience stored successfully")

# Test the update_q_network method (Note: This may not have a visible output, but you can check for errors)
dqn_agent.update_q_network(batch_size=32)

# Test the update_target_network method (Note: This may not have a visible output, but you can check for errors)
dqn_agent.update_target_network()

print("Testing complete.")

