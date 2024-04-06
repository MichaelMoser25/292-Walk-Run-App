from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

matplotlib.use('Agg')

from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os

# Validate form import
from wtforms.validators import InputRequired
import io

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, \
    RocCurveDisplay
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression

# Flask imports
from flask import Flask, render_template, send_file, redirect, url_for
from flask_wtf import FlaskForm
import os
import csv
import h5py
import numpy as np

# Create app instance
app = Flask(__name__)

# Create a secret key in app in order for form to show up in template
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Import jumping data for all members
member1_jumping_data = pd.read_csv("jump_joseph.csv")
member2_jumping_data = pd.read_csv("jump_michael.csv")
member3_jumping_data = pd.read_csv("jump_carl.csv")

# Save jumping data to a list
member_jumping_data = [member1_jumping_data, member2_jumping_data, member3_jumping_data]

# Import walking data for all members
member1_walking_data = pd.read_csv("walk_joseph.csv")
member2_walking_data = pd.read_csv("walk_michael.csv")
member3_walking_data = pd.read_csv("walk_carl.csv")

# Save walking data to a list
member_walking_data = [member1_walking_data, member2_walking_data, member3_walking_data]

# This function obtains the required 5-second window segments
def get_segments(data, column_name):
    data_windows = {}
    data_index = 0

    for i in range(0, data.shape[0], 500):
        if data.shape[0] - i >= 500:
            data_windows[data_index] = data[column_name][i:i + 500]
            data_index += 1

    return data_windows


member_jumping_data_windows = {}

# Obtain all 5-second windows for all 4 axes of jumping data
for member_index, data in enumerate(member_jumping_data):

    x_windows = get_segments(data, "Acceleration x (m/s^2)")
    y_windows = get_segments(data, "Acceleration y (m/s^2)")
    z_windows = get_segments(data, "Acceleration z (m/s^2)")
    abs_windows = get_segments(data, "Absolute acceleration (m/s^2)")

    member_jumping_data_windows[member_index] = {
        'x': x_windows,
        'y': y_windows,
        'z': z_windows,
        'abs': abs_windows
    }

member_walking_data_windows = {}

# Obtain all 5-second windows for all 4 axes of walking data
for member_index, data in enumerate(member_walking_data):

    x_windows = get_segments(data, "Acceleration x (m/s^2)")
    y_windows = get_segments(data, "Acceleration y (m/s^2)")
    z_windows = get_segments(data, "Acceleration z (m/s^2)")
    abs_windows = get_segments(data, "Absolute acceleration (m/s^2)")

    member_walking_data_windows[member_index] = {
        'x': x_windows,
        'y': y_windows,
        'z': z_windows,
        'abs': abs_windows
    }

# Concatenate jumping and walking windows together
merged_data = {**member_jumping_data_windows, **member_walking_data_windows}

def split_train_test(data_windows):

    windows_list = list(data_windows.values())

    train_windows, test_windows = train_test_split(
        windows_list, test_size=0.1, shuffle=True, random_state=42
    )

    return {'train': train_windows, 'test': test_windows}


member_train_test_data = {}

for member_index, axes_data in merged_data.items():
    member_train_test_data[member_index] = {}

    for axis, data_windows in axes_data.items():
        axis_train_test = split_train_test(data_windows)
        member_train_test_data[member_index][axis] = axis_train_test

# Step 3 - Visualization -> Note that this does not represent all figures shown in report

segment_number = 0
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes_labels = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
               'Absolute acceleration (m/s^2)']

for i, axis in enumerate(['x', 'y', 'z', 'abs']):
    segment = member_train_test_data[0][axis]['train'][segment_number]  # Get the segment for the current axis
    axs[i].plot(segment.values)  # Plot the segment
    axs[i].set_ylabel(axes_labels[i])  # Set the y-label for each subplot

axs[-1].set_xlabel('Sample Number')
axs[0].set_title('Walking Data')

plt.tight_layout()
plt.show()

segment_number = 5
fig, ax = plt.subplots(figsize=(10, 6))

# Define the axes labels for clarity in plotting
axes_labels = ['Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
               'Absolute acceleration (m/s^2)']

for i, axis in enumerate(['x', 'y', 'z', 'abs']):
    segment = member_train_test_data[0][axis]['train'][segment_number]
    ax.plot(segment.values, label=axes_labels[i])

ax.set_title('Jumping Data for Segment 5')
ax.set_xlabel('Sample Number')
ax.set_ylabel('Acceleration (m/s^2)')
ax.legend()
plt.show()

with h5py.File('accelerometer_data.h5', 'w') as hdf:

    member_identifiers = {0: 'joseph', 1: 'michael', 2: 'carl'}

    # Create 3 different groups for each team member with both the original "Jumping" and "Walking" datasets
    member1 = hdf.create_group('Joseph')
    member1.create_dataset('Walking Data', data=member1_walking_data)
    member1.create_dataset('Jumping Data', data=member1_jumping_data)

    member2 = hdf.create_group('Michael')
    member2.create_dataset('Walking Data', data=member2_walking_data)
    member2.create_dataset('Jumping Data', data=member2_jumping_data)

    member3 = hdf.create_group('Carl')
    member3.create_dataset('Walking Data', data=member3_walking_data)
    member3.create_dataset('Jumping Data', data=member3_jumping_data)

    dataset = hdf.create_group('Dataset')
    training_data = dataset.create_group('Training')
    testing_data = dataset.create_group('Testing')

    # Add training data for all members based on absolute acceleration windows
    for member_index in range(len(member_jumping_data)):
        member_id = member_identifiers[member_index]
        member_training_data = member_train_test_data[member_index]['abs']['train']
        for i, window_df in enumerate(member_training_data):
            window_array = window_df.to_numpy()
            training_data.create_dataset(f'{member_id}_window_{i}', data=window_array)

    # Add testing data for all members based on absolute acceleration windows
    for member_index in range(len(member_jumping_data)):
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
with h5py.File('accelerometer_data.h5', 'r') as hdf:

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
# ax.plot(x_input[:len(sma5_data_concat)], sma5_data_concat, linewidth=2, color='blue', label='SMA 5')
# ax.plot(x_input[:len(sma11_data_concat)], sma11_data_concat, linewidth=2, color='teal', label='SMA 11')
ax.plot(x_input[:len(sma21_data_concat)], sma21_data_concat, linewidth=2, color='magenta', label='SMA 21')
ax.set_title('Original Data (Purple) vs SMA 21 (Magenta)')
ax.set_xlabel('Data Point #')
ax.set_ylabel('Acceleration (m/s^2)')
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

# Build form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")


class Confusion(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Confusion Matrix")


# Create a home route
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    conf = Confusion()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # List the CSV files which already exist
        existing_files = [os.path.join(app.config['UPLOAD_FOLDER'], f)
                          for f in os.listdir(app.config['UPLOAD_FOLDER'])
                          if f.endswith('.csv')]

        # Output file path for concatenated data
        output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'concatenated_data.csv')

        # Concatenate the concatenate_data.csv with the uploaded data.csv file
        concatenate_csv(file_path, existing_files, output_file_path)

        return render_plot(filename)

    elif conf.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_confusion(filename)

    return render_template('index.html', form=form)


# Function to concatenate CSV files
def concatenate_csv(new_file_path, existing_file_paths, output_file_path):
    # Read the new CSV file
    new_data = pd.read_csv(new_file_path)

    # Read existing CSV files and concatenate
    existing_data = []
    for file_path in existing_file_paths:
        existing_data.append(pd.read_csv(file_path))
    existing_data.append(new_data)

    # Concatenate all dataframes
    concatenated_data = pd.concat(existing_data, ignore_index=True)

    # Save concatenated data to a new CSV file
    concatenated_data.to_csv(output_file_path, index=False)


# Route for success page for testing
@app.route('/success')
def success():
    return 'The CSV file was uploaded and concatenated successfully!'


def classify_activity(classify_data):
    walking_threshold = 9.1
    running_threshold = walking_threshold

    # Calculate the magnitude in the x,y,z plane
    classify_data['Magnitude'] = np.sqrt(
        classify_data['Acceleration x (m/s^2)'] ** 2 + classify_data['Acceleration y (m/s^2)'] ** 2 + classify_data['Acceleration z (m/s^2)'] ** 2)

    # Classify activity for the entire data
    classify_data['Activity'] = 'Unknown'
    classify_data.loc[classify_data['Magnitude'] >= walking_threshold, 'Activity'] = 'Walking'
    classify_data.loc[classify_data['Magnitude'] <= running_threshold, 'Activity'] = 'Jumping'

    return classify_data


def render_plot(filename):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    df = classify_activity(df)
    n_sample = len(df)
    x_input = np.arange(n_sample)

    # Display axis titles
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x_input, df['Magnitude'], label='Magnitude')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Magnitude')

    # Display the classified activity
    activity = df['Activity'].iloc[0]  # Assuming the activity classification is the same for the entire dataset
    ax.text(0.05, 0.95, f'Activity: {activity}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close(fig)

    if img_buffer:
        return send_file(img_buffer, mimetype='image/png')
    else:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/png')


def render_confusion(filename):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    df = classify_activity(df)
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png'))
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    return send_file(img_buffer, mimetype='image/png')

# Load the CSV file
df = pd.read_csv('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/walk_michael.csv')

# Setting data thresholds to determine activity
walking_threshold = 9.1  # For example, if the magnitude of acceleration is greater than 9.1 m/s^2, the data is considered to be walking
running_threshold = walking_threshold  # For example, if the magnitude of acceleration is less than walking_threshold, consider it as running

# Calculate the magnitude of acceleration
df['Magnitude'] = (df['Acceleration x (m/s^2)'] ** 2 + df['Acceleration y (m/s^2)'] ** 2 + df[
    'Acceleration z (m/s^2)'] ** 2) ** 0.5

# Classify each data point as walking or running based on thresholds
df['Activity'] = 'Unknown'
df.loc[df['Magnitude'] >= walking_threshold, 'Activity'] = 'Walking'
df.loc[df['Magnitude'] <= running_threshold, 'Activity'] = 'Jumping'

# Display the classification result
# print(df.columns)
# print(df.head())

data = df.iloc[:, 1:-1]
labels = df.iloc[:, -1]

# Assign 10% of the data to test the set
x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=0)

# Define a standard scaler to normalize inputs
scaler = StandardScaler()

# Define classifier and the pipline
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

# training
clf.fit(x_train, y_train)

# obtain predictions and probabilities
y_pred = clf.predict(x_test)
y_clf_prob = clf.predict_proba(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Calculate the recall of the model
recall = recall_score(y_test, y_pred, average='weighted')
# print("Recall:", recall)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png'))
plt.close()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'roc_curve.png'))
plt.close()

# Training the classifier
# convert sma21_data_concat to CSV
output_file_path = 'sma21_data_concat.csv'

# Save the DataFrame as a CSV file
sma21_data_concat.to_csv(output_file_path, index=False)
# df = pd.read_csv('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/concatenated_data.csv')
df = pd.read_csv('sma21_data_concat.csv')


# Display the classification result
# print(df.columns)
# print(df.head())

data = df.iloc[:, 1:-1]
labels = df.iloc[:, -1]

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

StandardScaler()
LogisticRegression(max_iter=10000)
PCA(n_components=2)

# Define the steps in the pipeline
steps = [
    ('scaler', StandardScaler()),   # Data normalization
    ('pca', PCA(n_components=2))    # PCA
]

# Create the pipeline
pca_pipe = Pipeline(steps)

# Apply the pipeline over X_train
X_train_pca = pca_pipe.fit_transform(x_train)

# Apply the same pipeline over X_test
X_test_pca = pca_pipe.transform(x_test)

# Define the logistic regression classifier
logistic_clf = LogisticRegression()

# Create a pipeline with logistic regression
clf = Pipeline([
    ('logistic', logistic_clf)
])

# Now the clf can be used for fitting and predicting

# Train clf with X_train_pca and y_train
clf.fit(X_train_pca, y_train)

# Obtain predictions for X_test_pca
y_pred_pca = clf.predict(X_test_pca)

# Create the decision boundary display using DecisionBoundaryDisplay()
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_pca, response_method="predict",
    xlabel='X1', ylabel='X2',
    alpha=0.5
)

# Map categorical labels to numeric values
label_mapping = {'Walking': 0, 'Jumping': 1}
y_train_numeric = y_train.map(label_mapping)

# Plot the scatter plot
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_numeric)

# Display model
plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'training_data.png'))

# Calculate accuracy score of the model using only 2 components of PCA
accuracy = accuracy_score(y_test, y_pred_pca)
# print('Accuracy after classifier: :', accuracy)

# ----------------------------------- used for testing, Empty the csv file
# # Create an empty DataFrame
# empty_df = pd.DataFrame()
#
# # Write the empty DataFrame to the concatenated file, overwriting its contents
# empty_df.to_csv('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/concatenated_data.csv', index=False)

if __name__ == '__main__':
    app.run(debug=True)
