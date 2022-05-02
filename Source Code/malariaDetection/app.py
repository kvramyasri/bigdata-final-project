from __future__ import division, print_function
import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/CNNmodel.h5'

# Loading the trained model
model = load_model(MODEL_PATH)
# Necessary to make everything ready to run on the GPU ahead of time
model.make_predict_function()
print('loaded the model successfully. Starting the web applications')


def model_predict(test_img, model):
    # adjust the size as the size should match the training data size
    img = cv2.resize(test_img, (50,50))
    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # predict the model
    preds = model.predict(img)
    pred = np.argmax(preds, axis=1)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        input_file = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        input_file.save(secure_filename("test.jpg"))
        test_img=cv2.imread("test.jpg")

        # Make prediction
        pred = model_predict(test_img, model)

        # Arrange the correct return according to the model.

        str1 = 'No Malaria Detected'
        str2 = 'Malaria Detected'
        if pred[0] == 0:
            return str1
        else:
            return str2
    return None


# run app locally
if __name__ == '__main__':
    app.run(debug=True)
