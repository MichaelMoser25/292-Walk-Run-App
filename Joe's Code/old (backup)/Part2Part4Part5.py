# This might need to be properly integrated into the main.py file, but I have tried to clean this up as much as I can
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# **We need to split member_data into "walking" and "jumping" using labels**

member1_data = pd.read_csv("jump_joseph.csv")
member2_data = pd.read_csv("jump_michael.csv")
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
        axis_train_test = split_train_test(data_windows)
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

# We are going to ignore x, y, and z acceleration for pre-processing and feature extraction and only focus on
# absolute acceleration as it makes no sense to consider 4 different axes for training and testing the model
training_windows_df = pd.DataFrame()
testing_windows_df = pd.DataFrame()

# List of dataframes for ease of accessing segmented windows
training_dataset_list = []

# Convert the HDF5 file into one big PD file that is still segmented into 5-second windows.
# This way we maintain the segmentation while still having access to PD features
with h5py.File('test.h5', 'r') as hdf:

    training_windows = hdf['Dataset/Training']

    testing_windows = hdf['Dataset/Testing']

    for window_indexes, window_name in enumerate(training_windows):

        window_data = training_windows[window_name][:]

        df_training = pd.DataFrame(window_data, columns=['Abs Accel (m/s^2)'])

        training_dataset_list.append(df_training)

        # training_windows_df = pd.concat([training_windows_df, df_training])

        # print(training_windows_df)

    for window_indexes in testing_windows:

        window_data = testing_windows[window_indexes][:]

        df_testing = pd.DataFrame(window_data, columns=['Abs Accel (m/s^2)'])

        testing_windows_df = pd.concat([testing_windows_df, df_testing])

        # print(testing_windows_df)

# Step 4: Pre-Processing

original_data = []
sma5_data = []
sma11_data = []
sma21_data = []

for window_dataframe in training_dataset_list:

    data = window_dataframe.iloc[:, 0]

    sma_5 = window_dataframe['Abs Accel (m/s^2)'].rolling(5).mean().dropna()

    sma_11 = window_dataframe['Abs Accel (m/s^2)'].rolling(11).mean().dropna()

    sma_21 = window_dataframe['Abs Accel (m/s^2)'].rolling(21).mean().dropna()

    original_data.append(data)

    sma5_data.append(sma_5)

    sma11_data.append(sma_11)

    sma21_data.append(sma_21)

# Concat used for graphing, ignore_index is used because otherwise points will be mapped to others with a horizontal
# line, Which makes the plots completely illegible
original_data_concat = pd.concat(original_data, ignore_index=True)
sma5_data_concat = pd.concat(sma5_data, ignore_index=True)
sma11_data_concat = pd.concat(sma11_data, ignore_index=True)
sma21_data_concat = pd.concat(sma21_data, ignore_index=True)

x_input = np.arange(len(original_data_concat))

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x_input, original_data_concat, linewidth=2, color='purple', label='Original')
ax.plot(x_input[:len(sma5_data_concat)], sma5_data_concat, linewidth=2, color='blue', label='SMA 5')
ax.plot(x_input[:len(sma11_data_concat)], sma11_data_concat, linewidth=2, color='teal', label='SMA 11')
ax.plot(x_input[:len(sma21_data_concat)], sma21_data_concat, linewidth=2, color='magenta', label='SMA 21')
ax.set_title("Original Data (Purple) vs SMA 5 (Blue) vs SMA 11 (Teal) vs SMA 21 (Magenta)")
ax.set_xlabel('Data Point #')
ax.set_ylabel('Amplitude')
plt.show()

print(sma21_data)

# Step 5: Feature Extraction

extracted_features = []

for smoothed_window_df in sma21_data:

    data = smoothed_window_df

    features = {
        'mean': smoothed_window_df.mean(),
        'min': smoothed_window_df.min(),
        'max': smoothed_window_df.max(),
        'median': smoothed_window_df.median(),
        'std_dev': smoothed_window_df.std(),
        'kurtosis': smoothed_window_df.kurtosis(),
        'variance': smoothed_window_df.var(),
        'skewness': smoothed_window_df.skew(),
        'sum': smoothed_window_df.sum(),
        'range': (smoothed_window_df.max() - smoothed_window_df.min()),
    }

    features_df = pd.DataFrame([features])

    extracted_features.append(features_df)

for i, features_df in enumerate(extracted_features):
    pd.set_option('display.max_columns', None)
    print(f"Features for Window {i+1}:\n", features_df)
    print("\n")









