import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
## Validate form import
from wtforms.validators import InputRequired

## pip3 install flask
## pip3 install flast_wtf wtforms

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
        file = form.file.data #Grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) #Save file
        return "File has been uploaded"

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
