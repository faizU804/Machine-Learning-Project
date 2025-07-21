import os
import pickle
import pandas as pd
import traceback

# Load the trained model
MODEL_PATH = os.path.join('output', 'best_model.pkl')

def test_model_loading():
    print(f"Testing model loading from {MODEL_PATH}")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file does not exist at {MODEL_PATH}")
            return None
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        print(f"Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        return None

def test_prediction(model):
    if model is None:
        print("Cannot test prediction as model failed to load")
        return
        
    print("\nTesting model prediction")
    try:
        # Create sample input data
        input_data = {
            'Suburb': ['Richmond'],
            'Rooms': [3],
            'Type': ['h'],
            'Method': ['S'],
            'SellerG': ['Biggin'],
            'Date': ['2017-03-04'],
            'Distance': [2.5],
            'Postcode': [3121],
            'Bedroom2': [3],
            'Bathroom': [1],
            'Car': [1],
            'Landsize': [202.0],
            'CouncilArea': ['Yarra'],
            'Lattitude': [-37.8136],
            'Longtitude': [144.9631],
            'Regionname': ['Northern Metropolitan'],
            'Propertycount': [4019]
        }
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame(input_data)
        print("Input DataFrame created:")
        print(input_df)
        
        # Make prediction
        print("Attempting prediction...")
        prediction = model.predict(input_df)
        print(f"Prediction successful: ${prediction[0]:,.2f}")
    except Exception as e:
        print(f"Error making prediction: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    model = test_model_loading()
    test_prediction(model) 