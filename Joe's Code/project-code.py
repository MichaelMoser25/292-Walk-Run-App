import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

dataset = pd.read_csv("walk_run.csv")
data = dataset.iloc[:, :]
data = pd.DataFrame(data)

sampling_rate = 100  # Phone accelerometer has 100 Hz sampling rate (samples per second) [REFERENCE]
window = 5  # 5-second window
samples = sampling_rate * window

number_of_windows = len(data) / samples

# Calculate the number of samples to include in the training set (90% of the data)
train_samples = int(number_of_windows * 0.9 * samples)

# Calculate the number of samples to include in the training set (90% of the data)
test_samples = int(number_of_windows * 0.1 * samples)

# Split the data into training and testing based on the calculated number of train_samples & test_samples
train_data = data.iloc[:train_samples]
test_data = data.iloc[:test_samples]

# Save the segmented data to new CSV files, if necessary
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

dataset = pd.read_csv("train_data.csv")
train_data = dataset.iloc[:, :]
train_data = pd.DataFrame(train_data)

dataset = pd.read_csv("test_data.csv")
test_data = dataset.iloc[:, :]
test_data = pd.DataFrame(test_data)

# *NOTE: This currently just dedicates the first 90% of the CSV file to training
# And the first 10% of the CSV file to testing. This is definitely not correct.

# This website is helpful for finding out if your HDF5 file works or not: https://myhdf5.hdfgroup.org/

with h5py.File('accel_data.h5', 'w') as hdf:

    G1 = hdf.create_group('/Member1')
    G1.create_dataset('dataset1', data=data)

    G2 = hdf.create_group('/Member2')
    G2.create_dataset('dataset2', data=data)

    G3 = hdf.create_group('/Member3')
    G3.create_dataset('dataset3', data=data)

    G4 = hdf.create_group('/dataset/Train')
    G4.create_dataset('segmented_training_data', data=train_data)

    G5 = hdf.create_group('/dataset/Test')
    G5.create_dataset('segmented_testing_data', data=test_data)


# Sample for length of data file
n_sample = len(data)
x_input = np.arange(n_sample)

# Apply rolling window filters
sma_5 = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
              'Absolute acceleration (m/s^2)']].rolling(5).mean()

sma_31 = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
               'Absolute acceleration (m/s^2)']].rolling(31).mean()

sma_51 = data[['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
               'Absolute acceleration (m/s^2)']].rolling(51).mean()

# Plot original data along with the SMAs for each axis
fig, ax = plt.subplots(4, 1, figsize=(10, 10))

# Axis titles (index 0 = x accel , index 1 = y accel, index 2 = z accel, index 3 = abs accel)
axis_titles = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
               'Absolute acceleration (m/s^2)']

# Plot for x-axis
ax[0].plot(x_input, data['Acceleration x (m/s^2)'][:n_sample], label='Original', linewidth=2)
ax[0].plot(x_input, sma_5['Acceleration x (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
ax[0].plot(x_input, sma_31['Acceleration x (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
ax[0].plot(x_input, sma_51['Acceleration x (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
ax[0].legend()
ax[0].set_title(axis_titles[0])
ax[0].set_xlabel('Sample')
ax[0].set_ylabel('Acceleration')

# Plot for y-axis
ax[1].plot(x_input, data['Acceleration y (m/s^2)'][:n_sample], label='Original', linewidth=2)
ax[1].plot(x_input, sma_5['Acceleration y (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
ax[1].plot(x_input, sma_31['Acceleration y (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
ax[1].plot(x_input, sma_51['Acceleration y (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
ax[1].legend()
ax[1].set_title(axis_titles[1])
ax[1].set_xlabel('Sample')
ax[1].set_ylabel('Acceleration')

# Plot for z-axis
ax[2].plot(x_input, data['Acceleration z (m/s^2)'][:n_sample], label='Original', linewidth=2)
ax[2].plot(x_input, sma_5['Acceleration z (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
ax[2].plot(x_input, sma_31['Acceleration z (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
ax[2].plot(x_input, sma_51['Acceleration z (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
ax[2].legend()
ax[2].set_title(axis_titles[2])
ax[2].set_xlabel('Sample')
ax[2].set_ylabel('Acceleration')

# Plot for abs-axis
ax[3].plot(x_input, data['Absolute acceleration (m/s^2)'][:n_sample], label='Original', linewidth=2)
ax[3].plot(x_input, sma_5['Absolute acceleration (m/s^2)'][:n_sample], label='SMA 5', linewidth=2)
ax[3].plot(x_input, sma_31['Absolute acceleration (m/s^2)'][:n_sample], label='SMA 31', linewidth=2)
ax[3].plot(x_input, sma_51['Absolute acceleration (m/s^2)'][:n_sample], label='SMA 51', linewidth=2)
ax[3].legend()
ax[3].set_title(axis_titles[3])
ax[3].set_xlabel('Sample')
ax[3].set_ylabel('Acceleration')

plt.tight_layout()  # Used to decompress the three axis graphs from each other (otherwise interference occurs)
plt.show()

# *NOTE: We still need to normalize the date in this step I think, I will do this later once we figure out HDF5
