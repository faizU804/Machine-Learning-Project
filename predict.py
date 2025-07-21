#!/usr/bin/env python
# Melbourne Housing Price Prediction - Inference Script

import pickle
import pandas as pd
import numpy as np
import os

def load_model():
    """Load the saved model from the output directory."""
    model_path = 'output/best_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run melbourne_housing_prediction.py first to train and save the model.")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully!")
    return model

def predict_price(model, input_data):
    """Make a prediction using the loaded model."""
    if not isinstance(input_data, pd.DataFrame):
        print("Error: Input data must be a pandas DataFrame.")
        return None
    
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def create_sample_input():
    """Create a sample input for demonstration."""
    # This sample includes all columns used during training
    sample = pd.DataFrame({
        'Suburb': ['Richmond'],
        'Rooms': [3],
        'Type': ['h'],  # h for house, u for unit, etc.
        'Method': ['S'],  # S for sold, etc.
        'SellerG': ['Biggin'],
        'Date': ['2017-03-04'],
        'Distance': [10.5],
        'Postcode': [3105],
        'Bedroom2': [3],
        'Bathroom': [1],
        'Car': [1],
        'Landsize': [600],
        'CouncilArea': ['Boroondara'],
        'Lattitude': [-37.85],
        'Longtitude': [145.01],
        'Regionname': ['Southern Metropolitan'],
        'Propertycount': [12000]
    })
    
    return sample

if __name__ == "__main__":
    # Load the trained model
    model = load_model()
    
    if model is None:
        exit(1)
    
    # Example usage with a sample input
    print("\n--- Sample Prediction ---")
    sample_input = create_sample_input()
    print("Sample input data:")
    print(sample_input[['Rooms', 'Type', 'Method', 'Distance', 'Propertycount']].head())  # Show only a few columns for readability
    
    # Make prediction
    prediction = predict_price(model, sample_input)
    
    if prediction is not None:
        print(f"\nPredicted Price: ${prediction[0]:,.2f}")
    
    # Interactive mode
    print("\n--- Interactive Prediction ---")
    print("Would you like to make a prediction with your own data? (yes/no)")
    response = input().lower()
    
    if response == 'yes':
        # This is a simplified example. You might want to add more validation and error handling
        input_data = {}
        
        # List of all features needed for prediction
        features = [
            'Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Date',
            'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
            'Landsize', 'CouncilArea', 'Lattitude', 'Longtitude',
            'Regionname', 'Propertycount'
        ]
        
        print("\nPlease enter the following information:")
        for feature in features:
            if feature in ['Rooms', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Propertycount']:
                value = int(input(f"{feature} (integer): "))
            elif feature in ['Distance', 'Landsize', 'Lattitude', 'Longtitude']:
                value = float(input(f"{feature} (decimal): "))
            else:
                value = input(f"{feature} (text): ")
            
            input_data[feature] = [value]
        
        # Create DataFrame and make prediction
        user_input = pd.DataFrame(input_data)
        user_prediction = predict_price(model, user_input)
        
        if user_prediction is not None:
            print(f"\nPredicted Price: ${user_prediction[0]:,.2f}")
    
    print("\nThank you for using the Melbourne Housing Price Predictor!") 