from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# Prediction threshold
THRESHOLD = 0.3

# Feature names for form and validation
FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst',
    'fractal_dimension_worst'
]

# Debug: Verify data.csv columns
try:
    data = pd.read_csv('data.csv')
    data_columns = data.drop(columns=['diagnosis'] if 'diagnosis' in data.columns else []).columns.tolist()
    if data_columns != FEATURES:
        print(f"Warning: data.csv columns do not match FEATURES. Expected: {FEATURES}, Got: {data_columns}")
except Exception as e:
    print(f"Error loading data.csv: {e}")

# Single Prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None or scaler is None:
        return render_template('index.html', error='Model or scaler not loaded. Please check files.')
    
    if request.method == 'POST':
        try:
            # Get form inputs
            inputs = []
            for feature in FEATURES:
                value = request.form.get(feature)
                if not value or not value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    return render_template('index.html', error=f'Invalid input for {feature}. All fields must be numeric.')
                inputs.append(float(value))
            inputs = np.array(inputs).reshape(1, -1)
            
            # Validate non-negative
            if np.any(inputs < 0):
                return render_template('index.html', error='All inputs must be non-negative.')
            
            # Scale inputs
            inputs_scaled = scaler.transform(inputs)
            
            # Predict
            prob = model.predict_proba(inputs_scaled)[:, 1][0]
            prediction = 1 if prob >= THRESHOLD else 0
            result = 'Malignant' if prediction == 1 else 'Benign'
            
            return render_template('result.html', result=result, probability=prob)
        except ValueError as ve:
            return render_template('index.html', error=f'Invalid input: {str(ve)}')
        except Exception as e:
            return render_template('index.html', error=f'An error occurred: {str(e)}')
    
    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)