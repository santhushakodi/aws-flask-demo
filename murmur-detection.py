import numpy as np
from flask import Flask, jsonify, request, redirect, session
import pickle
from flask import Flask, render_template
import mysql.connector
# import matlab.engine

# engine = matlab.engine.start_matlab()



# load the model from the file
with open('adaboost_classifier_with_spike_removed.pkl', 'rb') as f:
    model = pickle.load(f)


# def preprocessing(heart_signal):
#     pass


app = Flask(__name__)
app.secret_key = 'vortex123'

@app.route('/')
def home():
    return render_template('auth-signin.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def result():
    
    if not request.data:
        return jsonify(error="Please try again.")
    
    return jsonify(prediction="recieved the file")

@app.route('/search', methods=['POST'])
def murmur_show():
    data = request.get_json()
    pid = data['patient_id']
    print("patient id :", pid)

    # Process the data as required
    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    mycursor = mydb.cursor()
    query2 = "SELECT murmur FROM patients WHERE id=%s"
    data2 = (pid,)
    mycursor.execute(query2, data2)
    # fetch the result
    murmur = mycursor.fetchone()
    mycursor.close()
    mydb.close()
    
    result = {'message': 'Data processed successfully',
              'pid':pid,
              'murmur':murmur}
    return jsonify(result)

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['pwd']
        
        print(email, password)

        mydb = mysql.connector.connect(
            host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
            user="admin",
            password="admin123",
            database="demodb"
            )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = mycursor.fetchone()
        mycursor.close()
        print(user)

        if user and password == user[3]:
            session['email'] = email
            # return redirect('/dashboard')
            return render_template('index.html')
        else:
            error = 'Invalid username or password'
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/writedb', methods=['POST'])
def update():
    mydb = mysql.connector.connect(
        host="demo-database-1.cvs5fl0cptbn.eu-north-1.rds.amazonaws.com",
        user="admin",
        password="admin123",
        database="demodb"
        )
    mycursor = mydb.cursor()
    
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

@app.route('/matlab')
def run():
    result = engine.eval('disp("Hello, world!")')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
