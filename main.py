import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

from statsmodels.sandbox.regression.sympy_diff import df
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
## Validate form import
from wtforms.validators import InputRequired
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

from flask import Flask, render_template, send_file
from flask_wtf import FlaskForm
import os
import csv
import h5py
import numpy as np

## Create app instance
app = Flask(__name__)
## Create a secret key in app in order for form to show up in template
app.config['SECRET_KEY'] = 'supersecretkey'

app.config['UPLOAD_FOLDER'] = 'static/files'



## Build form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

## Create a home rout
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_plot(filename)
    return render_template('index.html', form=form)


def render_plot(filename):
    dataset = pd.read_csv("/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/walkrun.csv")
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

    img_buffer = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(img_buffer)
    img_buffer.seek(0)

    # Clear the plot to prevent it from being displayed again
    plt.close(fig)

    return send_file(img_buffer, mimetype='image/png')


# Read csv file
with open('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/Data.csv', "r", newline='') as csvfile:
    # Create reader object
    csv_reader = csv.reader(csvfile)

    # Convert CSV data to a list of lists
    dataset1 = list(csv_reader)


    # Write Uploaded data from csv file into hdf5
    with h5py.File('/Users/michaelmoser/ELEC292_Lab1/pythonProject/dataset.h5', 'w') as hdf:
        G1 = hdf.create_group('/Member1')
        G1.create_dataset('dataset1', data=dataset1)

        G2 = hdf.create_group('/dataset/Train')
        G2.create_dataset('dataset1', data=dataset1)

        G3 = hdf.create_group('/dataset/Test')
        G3.create_dataset('dataset1', data=dataset1)

    # Read from hdf5 file
    with h5py.File('/Users/michaelmoser/ELEC292_Lab1/pythonProject/dataset.h5', 'r') as hdf:
        dataset1 = hdf.get('/Member1/dataset1')[:]
        print(type(dataset1))
        my_array = np.array(dataset1)
        print(type(my_array))


    # Access dataset1
    with h5py.File('/Users/michaelmoser/ELEC292_Lab1/pythonProject/dataset.h5', 'r') as hdf:
        items = list(hdf.items())
        print(items)

        # Print Member 1 info
        G1 = hdf.get('/Member1')
        print(list(G1.items()))

        d1 = G1.get('dataset1')
        d1 = np.array(d1)
        print(d1.shape) #(1688,5)

    # Write csv file
    with open('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/time.csv', 'w', newline='') as csvfile:
        # Create csv writer object
        csv_writer = csv.writer(csvfile)

        # Write into the csv file
        for row in dataset1:
            csv_writer.writerow(row)

    # Load the CSV file
    df = pd.read_csv('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/walkrun.csv')

    # Assuming your CSV file contains columns like 'Acceleration x', 'Acceleration y', 'Acceleration z'
    # You may need to adjust these thresholds based on your data and experimentation
    walking_threshold = 9.8  # For example, if the magnitude of acceleration is less than 9.8 m/s^2, consider it as walking
    running_threshold = 15.0  # For example, if the magnitude of acceleration is greater than 15.0 m/s^2, consider it as running

    # Calculate the magnitude of acceleration
    # Assuming your CSV file has columns named 'X', 'Y', and 'Z' instead of 'Acceleration x', 'Acceleration y', and 'Acceleration z'
    df['Magnitude'] = (df['Acceleration x (m/s^2)'] ** 2 + df['Acceleration y (m/s^2)'] ** 2 + df['Acceleration z (m/s^2)'] ** 2) ** 0.5

    # Classify each data point as walking or running based on thresholds
    df['Activity'] = 'Unknown'
    df.loc[df['Magnitude'] <= walking_threshold, 'Activity'] = 'Walking'
    df.loc[df['Magnitude'] >= running_threshold, 'Activity'] = 'Running'

    # Display the classification result
    print(df.columns)
    print(df.head())




if __name__ == '__main__':
    app.run(debug=True)

