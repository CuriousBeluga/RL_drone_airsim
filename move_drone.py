import airsim
import time
import math

# Testing if python client to unreal engine 4 works

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Request API control
client.enableApiControl(True)
# Take off
client.takeoffAsync().join()

# Set the initial position of the quadcopter
initial_pose = client.simGetVehiclePose()
initial_position = initial_pose.position
# Set the target position (can use switch case to change the position)
target_position = airsim.Vector3r(initial_position.x_val+35, initial_position.y_val, initial_position.z_val)


movement_speed = 3
client.moveToPositionAsync(
    target_position.x_val,
    target_position.y_val,
    target_position.z_val,
    movement_speed
).join()

# Land the quadcopter
client.landAsync().join()

print("eagle has landed!")
