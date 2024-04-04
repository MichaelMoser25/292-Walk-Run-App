import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.model_selection import train_test_split

member1_data = pd.read_csv("walk_run.csv")
member2_data = pd.read_csv("random_data.csv")  # This is data that I got from a friend to test the code
# member3_data = pd.read_csv("walk_run_z")

# Need 6 separate data files (Walking & Jumping for each person) ???

# These for loops below will be defined as functions later when we actually get the rest of the code working

member_data = [member1_data, member2_data]

member_data_windows = {}

for member_index, data in enumerate(member_data):

    data_windows = {}
    data_index = 0

    for i in range(0, data.shape[0], 500):
        if data.shape[0] - i >= 500:
            data_windows[data_index] = (data["Absolute acceleration (m/s^2)"][i:i+500])
            data_index += 1

    # Store the segmented windows for this member
    member_data_windows[member_index] = data_windows

print(member_data_windows[0][0])  # Print statement to check that 5 second samples actually work
print(member_data_windows[1][0])  # Same as above ^

features_per_member = {}

for member, windows in member_data_windows.items():

    member_features = []

    for window_index in windows:

        # Access the actual window data using the index
        window_data = windows[window_index]

        # Calculate features for this window
        features = {
            'mean': window_data.mean(),
            'min': window_data.min(),
            'max': window_data.max(),
            "std_dev": window_data.std(),
            'median': window_data.median(),
        }

        # Create a DataFrame from the features dictionary
        features_df = pd.DataFrame([features])  # Encapsulate 'features' in a list to create a single-row DataFrame

        member_features.append(features_df)

    features_per_member[member] = pd.concat(member_features)

print(features_per_member[0])  # Prints features for all 75 windows for member 1
print(features_per_member[1])  # Prints features for all 61 windows for member 2

# Extract features -> Label as either walking/jumping -> Concatenate
# Concat all walking and jumping lists together
# Loop through list and extract features
# Do 90%, 10% split when done extracting the features (LATER)
# Use built in test_train_split function and set shuffle = TRUE

# with h5py.File('accel_data.h5', 'w') as hdf:
#
#     G1 = hdf.create_group('/Member1')
#     G1.create_dataset('dataset1', data=data)
#
#     # G2 = hdf.create_group('/Member2')
#     # G2.create_dataset('dataset2', data=member2_data)
#     #
#     # G3 = hdf.create_group('/Member3')
#     # G3.create_dataset('dataset3', data=member3_data)
#
#     # G4 = hdf.create_group('/dataset/Train')
#     # G4.create_dataset('segmented_training_data', data=train_data)
#     #
#     # G5 = hdf.create_group('/dataset/Test')
#     # G5.create_dataset('segmented_testing_data', data=test_data)

# # Sample for length of data file
# n_sample = len(data)
# x_input = np.arange(n_sample)
#
# # Apply rolling window filters
# sma_5 = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
#               'Absolute acceleration (m/s^2)']].rolling(5).mean()
#
# sma_31 = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
#                'Absolute acceleration (m/s^2)']].rolling(31).mean()
#
# sma_51 = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
#                'Absolute acceleration (m/s^2)']].rolling(51).mean()
#
# # Plot original data along with the SMAs for each axis
# fig, ax = plt.subplots(4, 1, figsize=(10, 10))
#
# # Axis titles (index 0 = x accel , index 1 = y accel, index 2 = z accel, index 3 = abs accel)
# axis_titles = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
#                'Absolute acceleration (m/s^2)']
#
# # Plot for x-axis
# ax[0].plot(x_input, data['Acceleration x (m/s^2)'][:n_sample], label='Original', linewidth=2)
# ax[0].plot(x_input, sma_5['Acceleration x (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
# ax[0].plot(x_input, sma_31['Acceleration x (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
# ax[0].plot(x_input, sma_51['Acceleration x (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
# ax[0].legend()
# ax[0].set_title(axis_titles[0])
# ax[0].set_xlabel('Sample')
# ax[0].set_ylabel('Acceleration')
#
# # Plot for y-axis
# ax[1].plot(x_input, data['Acceleration y (m/s^2)'][:n_sample], label='Original', linewidth=2)
# ax[1].plot(x_input, sma_5['Acceleration y (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
# ax[1].plot(x_input, sma_31['Acceleration y (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
# ax[1].plot(x_input, sma_51['Acceleration y (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
# ax[1].legend()
# ax[1].set_title(axis_titles[1])
# ax[1].set_xlabel('Sample')
# ax[1].set_ylabel('Acceleration')
#
# # Plot for z-axis
# ax[2].plot(x_input, data['Acceleration z (m/s^2)'][:n_sample], label='Original', linewidth=2)
# ax[2].plot(x_input, sma_5['Acceleration z (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
# ax[2].plot(x_input, sma_31['Acceleration z (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
# ax[2].plot(x_input, sma_51['Acceleration z (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
# ax[2].legend()
# ax[2].set_title(axis_titles[2])
# ax[2].set_xlabel('Sample')
# ax[2].set_ylabel('Acceleration')
#
# # Plot for abs-axis
# ax[3].plot(x_input, data['Absolute acceleration (m/s^2)'][:n_sample], label='Original', linewidth=2)
# ax[3].plot(x_input, sma_5['Absolute acceleration (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
# ax[3].plot(x_input, sma_31['Absolute acceleration (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
# ax[3].plot(x_input, sma_51['Absolute acceleration (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
# ax[3].legend()
# ax[3].set_title(axis_titles[3])
# ax[3].set_xlabel('Sample')
# ax[3].set_ylabel('Acceleration')
#
# plt.tight_layout()  # Used to decompress the three axis graphs from each other (otherwise interference occurs)
# plt.show()
#
# # *NOTE: We still need to normalize the date in this step I think, I will do this later once we figure out HDF5
