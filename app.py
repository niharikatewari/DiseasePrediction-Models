import joblib
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import gunicorn

filename = 'modelsvm.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('modelsvm.pkl', 'rb'))

app = Flask(__name__,template_folder='Templates')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict-value', methods=['POST'])
def predict_value():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer!"
    else:
        res_val = "a low risk of Breast Cancer :)"

    return render_template('breastcancerresult.html', prediction_text='Patient has {}'.format(res_val))


##################################################################################

if __name__ == "__main__":
    app.run(debug=True)
