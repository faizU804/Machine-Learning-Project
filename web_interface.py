#!/usr/bin/env python
# Melbourne Housing Price Prediction - Web Interface

import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import traceback
from datetime import datetime

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Load the trained model
MODEL_PATH = os.path.join('output', 'best_model.pkl')
ACCURACY_PATH = os.path.join('output', 'model_accuracy.txt')
SELECTED_FEATURES_PATH = os.path.join('output', 'selected_features.txt')

def load_model():
    try:
        print(f"Attempting to load model from {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file does not exist at {MODEL_PATH}")
            return None
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        return None

def load_accuracy():
    try:
        with open(ACCURACY_PATH, 'r') as f:
            lines = f.readlines()
            r2 = float(lines[0].split(':')[1].strip())
            rmse = lines[1].split(':')[1].strip()
            return r2, rmse
    except Exception as e:
        print(f"Error loading accuracy: {e}")
        return None, None

def load_selected_features():
    try:
        with open(SELECTED_FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        return features
    except Exception as e:
        print(f"Error loading selected features: {e}")
        return []

def preprocess_input_data(input_data):
    """
    Preprocess the input data to match the format expected by the model.
    """
    # Format the date correctly if provided as string
    if 'Date' in input_data and input_data['Date'][0]:
        try:
            # Try to parse the date to ensure it's in the right format
            date_obj = datetime.strptime(input_data['Date'][0], '%Y-%m-%d')
            input_data['Date'] = [date_obj.strftime('%Y-%m-%d')]
        except Exception as e:
            print(f"Error formatting date: {e}")
    
    # Make sure numeric values are correct types
    numeric_columns = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 
                       'Landsize', 'Lattitude', 'Longtitude', 'Propertycount']
    
    for col in numeric_columns:
        if col in input_data:
            try:
                input_data[col] = [float(input_data[col][0])]
            except Exception as e:
                print(f"Error converting {col} to float: {e}")
    
    return input_data

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            model = load_model()
            r2, rmse = load_accuracy()
            selected_features = load_selected_features()
            if model is None:
                return render_template('error.html', error_message='Could not load the prediction model. Please make sure you have run the training script.')
            # Get form data (only selected features)
            rooms = int(request.form.get('rooms'))
            distance = float(request.form.get('distance'))
            postcode = int(request.form.get('postcode'))
            bedrooms = int(request.form.get('bedrooms'))
            bathrooms = int(request.form.get('bathrooms'))
            landsize = float(request.form.get('landsize'))
            latitude = float(request.form.get('latitude'))
            longitude = float(request.form.get('longitude'))
            property_count = int(request.form.get('property_count'))
            property_type = request.form.get('property_type')
            seller = request.form.get('seller')
            region = request.form.get('region')
            # Prepare input data for prediction (only selected features)
            input_data = {
                'Rooms': [rooms],
                'Distance': [distance],
                'Postcode': [postcode],
                'Bedroom2': [bedrooms],
                'Bathroom': [bathrooms],
                'Landsize': [landsize],
                'Lattitude': [latitude],
                'Longtitude': [longitude],
                'Propertycount': [property_count],
                'Type': [property_type],
                'SellerG': [seller],
                'Regionname': [region]
            }
            input_df = pd.DataFrame(input_data)
            prediction = model.predict(input_df)
            predicted_price = prediction[0]
            formatted_price = f"${predicted_price:,.2f}"
            form_data = {
                'rooms': rooms,
                'distance': distance,
                'postcode': postcode,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'landsize': landsize,
                'latitude': latitude,
                'longitude': longitude,
                'property_count': property_count,
                'property_type': property_type,
                'seller': seller,
                'region': region
            }
            return render_template('result.html', predicted_price=formatted_price, form_data=form_data, model_accuracy=r2, model_rmse=rmse)
        except Exception as e:
            error_message = f'Error making prediction: {str(e)}'
            print(error_message)
            print(traceback.format_exc())
            return render_template('error.html', error_message=error_message)
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}")
        print("Please run melbourne_housing_prediction.py first to train and save the model.")
    else:
        print(f"Model file found at {MODEL_PATH}")
    
    # Run the app
    app.run(debug=True) 