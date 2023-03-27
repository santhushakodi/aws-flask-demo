import numpy as np
from flask import Flask, jsonify, request
import pickle
from flask import Flask, render_template




# load the model from the file
with open('adaboost_classifier_with_spike_removed.pkl', 'rb') as f:
    model = pickle.load(f)


# def preprocessing(heart_signal):
#     pass


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def result():

    if 'message' in request.get_json():
        res = {"message":"good"}
        return jsonify(res)
    
    if 'feature' not in request.files:
        print()
        return jsonify(error="Please try again.")
    
    


    file = request.files.get('feature')
    feature  = np.load(file)
    result = model.predict(feature)
    result = result.tolist()
    return jsonify(prediction=result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')