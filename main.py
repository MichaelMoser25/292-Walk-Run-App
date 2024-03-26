
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



# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# ## Extract data from csv file
# dataset = pd.read_csv('test.csv')

# ## Seperate labels from data
# data = dataset.iloc[:, :-1]
# labels = dataset.iloc[:, -1]

# ## Plot a 4 by 4 grid
# fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(20, 10))

# data.hist(ax=ax.flatten()[0:13]) ## Data has 13 columns

# ## Render plot
# fig.tight_layout()
# plt.show()



# ## Seperate labels from data
# dataset = pd.read_csv("unclean-wine-quality.csv")
# dataset = dataset.iloc[:, 1:-1]

# ## Clean the data
# nanIndices = np.where(pd.isnull(dataset))
# print("The indices of NaNs are:\n ", nanIndices)
# numNans = dataset.isna().sum().sum()
# print("The total number of NaNs is:", numNans)

# print("\n")

# dashIndices = np.where(dataset == "-")
# print("The indices of \"-\" are:\n", dashIndices)
# numDashes = np.sum(dataset.values == "-").sum()
# print("The total number of \"-\" is: ", numDashes)

# dataset.mask(dataset == "-", other=np.nan, inplace=True)
# dataset = dataset.astype('float64')


# ## Replace NaN values with constants
# fill_values = {
#     'fixed acidity': 0,
#     'volatile acidity': 0,
#     'citric acid': 0,
#     'residual sugar': 0,
#     'chlorides': 1,
#     'free sulfur dioxide': 0,
#     'total sulfur dioxide': 0,
#     'density': 0,
#     'pH': 1,
#     'sulphates': 1,
#     'alcohol': 0
# }

# dataset.fillna(fill_values, inplace=True)
# numNans = dataset.isna().sum().sum()
# print("The number of NaNs is: ", numNans)


# ## For example, if a NaN exists under the column ‘pH’, it should be replaced with 1. After replacing
# ## all NaNs with the specified values, print the total number of NaNs in the dataset again.


# ## Sample and hold filling
# dataset.fillna(method='ffill', inplace=True)
# print("[16, 0]: ", dataset.iloc[16, 0])
# print("[17, 0]: ", dataset.iloc[17, 0])

# ## Linear interpolation
