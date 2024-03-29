import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("walk_run.csv")
data = dataset.iloc[:, :]
data = pd.DataFrame(data)

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
