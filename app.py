from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=["POST"])
def predict():
    num_features = [float(x) for x in request.form.values()]
    total_agg = num_features[5] + num_features[6]
    wc_ratio = num_features[3] / num_features[0]
    wb_ratio = num_features[3] / ( num_features[0] + num_features[1] + num_features[2] )
    concrete = num_features[0] + num_features[1] + num_features[2] + num_features[3] + num_features[4] + num_features[5] + num_features[6]
    
    num_features.append(total_agg)
    num_features.append(wc_ratio)
    num_features.append(wb_ratio)
    num_features.append(concrete)

    num_features.remove(num_features[9])

    final_features = [np.array(num_features)]

    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('home.html', prediction_text='Compressive Strength of Concrete = {} MPa'.format(output))


if __name__ == "__main__" :
    app.run(debug=True)
