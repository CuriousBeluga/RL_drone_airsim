import airsim
import time
import math


# Testing if python client to unreal engine 4 works


def move_agent(x, y, z, speed, client):
    # move agent
    print(f'Moving agent by ({x},{y},{z})')
    current_pose = client.simGetVehiclePose()
    curr_position = current_pose.position
    target_position = airsim.Vector3r(curr_position.x_val + x, curr_position.y_val + y, curr_position.z_val + z)
    client.moveToPositionAsync(target_position.x_val, target_position.y_val, target_position.z_val, speed).join()


def no_obstacle_run():
    # Connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()

    # Request API control
    client.enableApiControl(True)
    # Take off
    client.takeoffAsync().join()
    movement_speed = 3
    for i in range(0,8):
        move_agent(3, 0, 0, movement_speed, client)

    # Land the quadcopter
    client.landAsync().join()

    print("eagle has landed!")


def one_obstacle_run():
    # Connect to the AirSim simulator

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.reset()
    # Request API control
    client.enableApiControl(True)
    # Take off
    client.takeoffAsync().join()

    movement_speed = 3

    move_agent(1, 0, 0, movement_speed, client)
    move_agent(3, 0, 0, movement_speed, client)
    move_agent(6, 0, 0, movement_speed, client)
    move_agent(6, 2.5, -2, movement_speed, client)
    move_agent(6, 0, -2, movement_speed, client)
    move_agent(9, 0, -2, movement_speed, client)

    # Land the quadcopter
    client.landAsync().join()

    print("eagle has landed!")


def multi_obstacle_run():
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
    target_position = airsim.Vector3r(initial_position.x_val + 35, initial_position.y_val, initial_position.z_val)

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


# ---------------------------------------------------
# uncomment to run the obstacle course that you want
no_obstacle_run()
# one_obstacle_run()
# multi_obstacle_run()
