from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')  # Ensure this matches your actual model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect inputs from form
            features = [
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['chest_pain']),
                float(request.form['resting_bp']),
                float(request.form['cholesterol']),
                float(request.form['fasting_bs']),
                float(request.form['restecg']),
                float(request.form['max_hr']),
                float(request.form['exercise_angina']),
                float(request.form['st_depression']),
                float(request.form['st_slope']),
                float(request.form['major_vessels']),
                float(request.form['thalassemia'])
            ]

            # Reshape and predict
            final_input = np.array(features).reshape(1, -1)
            prediction = model.predict(final_input)

            result = "You have High Risk of Heart Disease" if prediction[0] == 1 else "You have Low Risk of Heart Disease"
            return render_template('index.html', prediction_text=f'Prediction Result: {result}')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error during prediction: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
