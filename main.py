
from sklearn.pipeline import make_pipeline
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay

matplotlib.use('Agg')

from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os

## Validate form import
from wtforms.validators import InputRequired
import io

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, \
    RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression



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

class Confusion(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Confusion Matrix")

## Create a home rout
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    conf = Confusion()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_plot(filename)
    elif conf.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_confusion(filename)

    return render_template('index.html', form=form)


def classify_activity(data):
    walking_threshold = 5
    running_threshold = 15.0

    jumping_threshold = 5

    data['Magnitude'] = np.sqrt(data['Acceleration x (m/s^2)'] ** 2 + data['Acceleration y (m/s^2)'] ** 2 + data['Acceleration z (m/s^2)'] ** 2)

    # Classify activity for the entire data
    data['Activity'] = 'Unknown'
    # data.loc[data['Magnitude'] <= walking_threshold, 'Activity'] = 'Walking'
    # data.loc[data['Magnitude'] >= running_threshold, 'Activity'] = 'Running'
    data.loc[data['Magnitude'] >= jumping_threshold, 'Activity'] = 'Jumping'

    return data
def render_plot(filename):
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    df = classify_activity(df)
    n_sample = len(df)
    x_input = np.arange(n_sample)

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

    return send_file(img_buffer, mimetype='image/png')
    # return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/png')

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
    df = pd.read_csv('/Users/michaelmoser/ELEC292_Lab1/pythonProject/Static/files/jump_michael.csv')

    # Assuming your CSV file contains columns like 'Acceleration x', 'Acceleration y', 'Acceleration z'
    # You may need to adjust these thresholds based on your data and experimentation
    walking_threshold = 9.8  # For example, if the magnitude of acceleration is less than 9.8 m/s^2, consider it as walking
    running_threshold = 9  # For example, if the magnitude of acceleration is greater than 15.0 m/s^2, consider it as running

    # Calculate the magnitude of acceleration
    # Assuming your CSV file has columns named 'X', 'Y', and 'Z' instead of 'Acceleration x', 'Acceleration y', and 'Acceleration z'
    df['Magnitude'] = (df['Acceleration x (m/s^2)'] ** 2 + df['Acceleration y (m/s^2)'] ** 2 + df['Acceleration z (m/s^2)'] ** 2) ** 0.5

    # Classify each data point as walking or running based on thresholds
    df['Activity'] = 'Unknown'
    df.loc[df['Magnitude'] <= walking_threshold, 'Activity'] = 'Walking'
    df.loc[df['Magnitude'] >= running_threshold, 'Activity'] = 'Jumping'

    # Display the classification result
    print(df.columns)
    print(df.head())


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

    #training
    clf.fit(x_train, y_train)

    # obtain predictions and probabilities
    y_pred = clf.predict(x_test)
    y_clf_prob = clf.predict_proba(x_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate the recall of the model
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)

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


if __name__ == '__main__':
    app.run(debug=True)
