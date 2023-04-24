import numpy as np
from flask import Flask, jsonify, request
import pickle
from flask import Flask, render_template
import mysql.connector



# load the model from the file
with open('adaboost_classifier_with_spike_removed.pkl', 'rb') as f:
    model = pickle.load(f)


# def preprocessing(heart_signal):
#     pass


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def result():

    # if 'message' in request.get_json():
    #     res = {"message":"good"}
    #     return jsonify(res)
    
    if 'feature' not in request.files:
        print()
        return jsonify(error="Please try again.")
    

    file = request.files.get('feature')
    feature  = np.load(file)
    result = model.predict(feature)
    result = result.tolist()
    return jsonify(prediction=result)

@app.route('/writedb', methods=['POST'])
def update():
    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    
    data = request.get_json()
    id = data.get("id")
    murmur = data.get("murmur")
    print(id,murmur)
    mycursor = mydb.cursor()

    
    query1 = "INSERT INTO patients (id, murmur) VALUES (%s, %s)"
    data = (id, murmur)
    mycursor.execute(query1, data)
    # commit the transaction
    mydb.commit()

    query2 = "SELECT * FROM patients WHERE id=%s"
    data2 = (id,)
    mycursor.execute(query2, data2)
    # fetch the result
    result = mycursor.fetchone()
    mycursor.close()
    mydb.close()

    resp = {"id":result[0],"mumur":result[1]}
    return jsonify(resp)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
