# This might need to be properly integrated into the main.py file, but I have tried to clean this up as much as I can

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split

# **We need to split member_data into "walking" and "jumping" using labels**

member1_data = pd.read_csv("walk_run.csv")
member2_data = pd.read_csv("random_data.csv")
# member3_data = pd.read_csv("walk_run_z")

member_data = [member1_data, member2_data]

# This function obtains the required 5-second window segments
def get_segments(data, column_name):
    data_windows = {}
    data_index = 0

    for i in range(0, data.shape[0], 500):
        if data.shape[0] - i >= 500:
            data_windows[data_index] = data[column_name][i:i + 500]
            data_index += 1

    return data_windows


member_data_windows = {}

for member_index, data in enumerate(member_data):

    x_windows = get_segments(data, "Acceleration x (m/s^2)")
    y_windows = get_segments(data, "Acceleration y (m/s^2)")
    z_windows = get_segments(data, "Acceleration z (m/s^2)")
    abs_windows = get_segments(data, "Absolute acceleration (m/s^2)")

    member_data_windows[member_index] = {
        'x': x_windows,
        'y': y_windows,
        'z': z_windows,
        'abs': abs_windows
    }

# Visualization (WIP)
# for i in range(0, 5):
#     member_index = 0
#     x_window = member_data_windows[member_index]['x'][i]
#     y_window = member_data_windows[member_index]['y'][i]
#     z_window = member_data_windows[member_index]['z'][i]
#     abs_window = member_data_windows[member_index]['abs'][i]
#
#     # Setup subplots for one member
#     fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
#
#     # X Axis Acceleration
#     axs[0].plot(x_window, color='red')
#     axs[0].set_ylabel('Accel (m/s^2)')
#     axs[0].set_title('X Axis Acceleration')
#
#     # Y Axis Acceleration
#     axs[1].plot(y_window, color='blue')
#     axs[1].set_ylabel('Accel (m/s^2)')
#     axs[1].set_title('Y Axis Acceleration')
#
#     # Z Axis Acceleration
#     axs[2].plot(z_window, color='green')
#     axs[2].set_ylabel('Accel (m/s^2)')
#     axs[2].set_title('Z Axis Acceleration')
#
#     # Absolute Acceleration
#     axs[3].plot(abs_window, color='purple')
#     axs[3].set_ylabel('Accel (m/s^2)')
#     axs[3].set_xlabel('Time (s)')
#     axs[3].set_title('Absolute Acceleration')
#
#     plt.tight_layout()  # Adjust layout to make room for titles/labels
#     plt.show()

def split_train_test(data_windows):

    windows_list = list(data_windows.values())

    train_windows, test_windows = train_test_split(
        windows_list, test_size=0.1, shuffle=True, random_state=42
    )

    return {'train': train_windows, 'test': test_windows}


member_train_test_data = {}

for member_index, axes_data in member_data_windows.items():
    member_train_test_data[member_index] = {}

    for axis, data_windows in axes_data.items():
        # Split the windows for the current axis into training and testing sets
        axis_train_test = split_train_test(data_windows)

        # Store the split data
        member_train_test_data[member_index][axis] = axis_train_test

with h5py.File('test.h5', 'w') as hdf:

    member_identifiers = {0: 'joseph', 1: 'michael', 2: 'carl'}

    # Create 3 different groups for each team member with both the original "Jumping" and "Walking" datasets
    member1 = hdf.create_group('Joseph')
    member1.create_dataset('Walking Data', data=member1_data)
    member1.create_dataset('Jumping Data', data=member1_data)

    member2 = hdf.create_group('Michael')
    member2.create_dataset('Walking Data', data=member2_data)
    member2.create_dataset('Jumping Data', data=member2_data)

    member3 = hdf.create_group('Carl')
    member3.create_dataset('Walking Data', data=member2_data)
    member3.create_dataset('Jumping Data', data=member2_data)

    dataset = hdf.create_group('Dataset')
    training_data = dataset.create_group('Training')
    testing_data = dataset.create_group('Testing')

    # Add training data for all members based on absolute acceleration windows
    for member_index in range(len(member_data)):
        member_id = member_identifiers[member_index]
        member_training_data = member_train_test_data[member_index]['abs']['train']
        for i, window_df in enumerate(member_training_data):
            window_array = window_df.to_numpy()
            training_data.create_dataset(f'{member_id}_window_{i}', data=window_array)

    # Add testing data for all members based on absolute acceleration windows
    for member_index in range(len(member_data)):
        member_id = member_identifiers[member_index]
        member_testing_data = member_train_test_data[member_index]['abs']['test']
        for i, window_df in enumerate(member_testing_data):
            window_array = window_df.to_numpy()
            testing_data.create_dataset(f'{member_id}_window_{i}', data=window_array)

    # Note that although window names may be repeated from both testing and training sets, they are not the same windows
    # i.e. "Window 6" in the training group does not correspond to "Window6" in the testing group, they are separate.
    # https://myhdf5.hdfgroup.org/ Can be used to verify this

