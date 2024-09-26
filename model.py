from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model and the scaler
model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = float(request.form.get('age'))
        sex = request.form.get('sex')
        education = float(request.form.get('education'))
        dominant_hand = request.form.get('dominant_hand')
        visits = float(request.form.get('visits'))
        ses = float(request.form.get('ses'))

        # Additional fields for the new features
        mr_delay = float(request.form.get('mr_delay'))
        asf = float(request.form.get('asf'))
        mmse = float(request.form.get('mmse'))
        cdr = float(request.form.get('cdr'))
        etiv = float(request.form.get('etiv'))
        nwbv = float(request.form.get('nwbv'))

        # Map 'sex' and 'dominant_hand' to numeric values
        sex_encoded = 1 if sex == 'M' else 0  # 1 for Male, 0 for Female
        dominant_hand_encoded = 1 if dominant_hand == 'Right' else 0  # 1 for Right, 0 for Left

        # Prepare the feature array for prediction
        features = np.array([[age, sex_encoded, education, dominant_hand_encoded, visits, ses, mr_delay, asf,
                              mmse, cdr, etiv, nwbv]])

        # Scale the features using the same scaler from training
        features_scaled = scaler.transform(features)

        # Make prediction using the trained model
        prediction = model.predict(features_scaled)[0]

        # Render the result back to the page
        return render_template('index.html', prediction=prediction)

    except Exception as e:
        # Handle any errors and display them on the page
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Get the port from the environment (Render sets it automatically)
    port = int(os.environ.get('PORT', 5000))  # Fallback to 5000 if no port is found
    app.run(host='0.0.0.0', port=port, debug=True)
