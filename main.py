
# import pandas as pd
# import numpy as np
# import csv
# import h5py
#
#
# from flask import Flask, render_template
# from flask_wtf import FlaskForm
# from statsmodels.sandbox.regression.sympy_diff import df
# from wtforms import FileField, SubmitField
# from werkzeug.utils import secure_filename
# import os
# ## Validate form import
# from wtforms.validators import InputRequired
#
# ## pip3 install flask
# ## pip3 install flast_wtf wtforms
#
# ## Create app instance
# app = Flask(__name__)
# ## Create a secret key in app in order for form to show up in template
# app.config['SECRET_KEY'] = 'supersecretkey'
#
# app.config['UPLOAD_FOLDER'] = 'static/files'
#
# ## Build form
# class UploadFileForm(FlaskForm):
#     file = FileField("File", validators=[InputRequired()])
#     submit = SubmitField("Upload File")
#
# ## Create a home rout
# @app.route('/', methods=['GET', 'POST'])
# @app.route('/home', methods=['GET', 'POST'])
# def home():
#     form = UploadFileForm()
#     if form.validate_on_submit():
#         file = form.file.data #Grab the file
#         file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) #Save file
#         return "File has been uploaded"
#
#     return render_template('index.html', form=form)
#
#
#
#
# # Read csv file
# with open('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/Data.csv', "r", newline='') as csvfile:
#     # Create reader object
#     csv_reader = csv.reader(csvfile)
#
#     # Convert CSV data to a list of lists
#     dataset1 = list(csv_reader)
#
#
#     # Write Uploaded data from csv file into hdf5
#     with h5py.File('/Users/michaelmoser/ELEC292_Lab1/pythonProject/dataset.h5', 'w') as hdf:
#         G1 = hdf.create_group('/Member1')
#         G1.create_dataset('dataset1', data=dataset1)
#
#         G2 = hdf.create_group('/dataset/Train')
#         G2.create_dataset('dataset1', data=dataset1)
#
#         G3 = hdf.create_group('/dataset/Test')
#         G3.create_dataset('dataset1', data=dataset1)
#
#     # Read from hdf5 file
#     with h5py.File('/Users/michaelmoser/ELEC292_Lab1/pythonProject/dataset.h5', 'r') as hdf:
#         dataset1 = hdf.get('/Member1/dataset1')[:]
#         print(type(dataset1))
#         my_array = np.array(dataset1)
#         print(type(my_array))
#
#
#     # Access dataset1
#     with h5py.File('/Users/michaelmoser/ELEC292_Lab1/pythonProject/dataset.h5', 'r') as hdf:
#         items = list(hdf.items())
#         print(items)
#
#         # Print Member 1 info
#         G1 = hdf.get('/Member1')
#         print(list(G1.items()))
#
#         d1 = G1.get('dataset1')
#         d1 = np.array(d1)
#         print(d1.shape) #(1688,5)
#
#     # Write csv file
#     with open('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/time.csv', 'w', newline='') as csvfile:
#         # Create csv writer object
#         csv_writer = csv.writer(csvfile)
#
#         # Write into the csv file
#         for row in dataset1:
#             csv_writer.writerow(row)
#
# # Load dataset
# dataset = pd.read_csv('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/Data.csv')
#
# # Plot the data
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # Plot each column
# ax.plot(dataset['Time (s)'], dataset['Linear Acceleration x (m/s^2)'], label='Linear Acceleration x')
# ax.plot(dataset['Time (s)'], dataset['Linear Acceleration y (m/s^2)'], label='Linear Acceleration y')
# ax.plot(dataset['Time (s)'], dataset['Linear Acceleration z (m/s^2)'], label='Linear Acceleration z')
# ax.plot(dataset['Time (s)'], dataset['Absolute acceleration (m/s^2)'], label='Absolute acceleration')
#
# # Set labels and title
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Acceleration (m/s^2)')
# ax.set_title('Acceleration vs Time')
# ax.legend()
#
# # Show plot
# plt.show()
#
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
#
#
#
#
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

from flask import Flask, render_template, send_file
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import csv
import h5py
import numpy as np

# Create app instance
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Define upload form
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

# Route for home page
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

# Function to render the plot
def render_plot(filename):
    # Load dataset
    dataset = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dataset['Time (s)'], dataset['Linear Acceleration x (m/s^2)'], label='Linear Acceleration x')
    ax.plot(dataset['Time (s)'], dataset['Linear Acceleration y (m/s^2)'], label='Linear Acceleration y')
    ax.plot(dataset['Time (s)'], dataset['Linear Acceleration z (m/s^2)'], label='Linear Acceleration z')
    ax.plot(dataset['Time (s)'], dataset['Absolute acceleration (m/s^2)'], label='Absolute acceleration')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration (m/s^2)')
    ax.set_title('Acceleration vs Time')
    ax.legend()

    # Save plot to a temporary buffer
    img_buffer = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(img_buffer)
    img_buffer.seek(0)

    # Clear the plot to prevent it from being displayed again
    plt.close(fig)

    return send_file(img_buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
