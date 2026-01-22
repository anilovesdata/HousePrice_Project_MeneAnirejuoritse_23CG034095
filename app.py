from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessing objects
MODEL_PATH = 'model/house_price_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODER_PATH = 'model/neighborhood_encoder.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✓ Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model, scaler, encoder = None, None, None

# Get neighborhood options from encoder
neighborhoods = encoder.classes_.tolist() if encoder else []

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files exist.'
            }), 500
        
        # Get input data from form
        overall_qual = float(request.form.get('overall_qual', 0))
        gr_liv_area = float(request.form.get('gr_liv_area', 0))
        total_bsmt_sf = float(request.form.get('total_bsmt_sf', 0))
        garage_cars = float(request.form.get('garage_cars', 0))
        year_built = float(request.form.get('year_built', 0))
        neighborhood = request.form.get('neighborhood', '')
        
        # Validate inputs
        if not all([overall_qual, gr_liv_area, year_built, neighborhood]):
            return jsonify({
                'error': 'Please fill in all required fields.'
            }), 400
        
        # Encode neighborhood
        try:
            neighborhood_encoded = encoder.transform([neighborhood])[0]
        except ValueError:
            return jsonify({
                'error': f'Invalid neighborhood: {neighborhood}'
            }), 400
        
        # Create feature array
        # Order: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, YearBuilt, Neighborhood_encoded
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Format prediction
        predicted_price = f"${prediction:,.2f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'raw_price': float(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoder_loaded': encoder is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Please ensure the model directory contains all required files.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)