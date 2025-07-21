import os
import pickle
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from .forms import PredictionForm

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                         'output', 'best_model.pkl')

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def home(request):
    """Home page view with the prediction form"""
    form = PredictionForm()
    return render(request, 'predictor/home.html', {'form': form})

def predict(request):
    """Process the form data and make predictions"""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Load the model
            model = load_model()
            if model is None:
                return render(request, 'predictor/error.html', {
                    'error_message': 'Could not load the prediction model. Please make sure you have run the training script.'
                })
            
            # Prepare input data for prediction
            input_data = {
                'Suburb': [form.cleaned_data['suburb']],
                'Rooms': [form.cleaned_data['rooms']],
                'Type': [form.cleaned_data['property_type']],
                'Method': [form.cleaned_data['method']],
                'SellerG': [form.cleaned_data['seller']],
                'Date': [form.cleaned_data['date'].strftime('%Y-%m-%d')],
                'Distance': [form.cleaned_data['distance']],
                'Postcode': [form.cleaned_data['postcode']],
                'Bedroom2': [form.cleaned_data['bedrooms']],
                'Bathroom': [form.cleaned_data['bathrooms']],
                'Car': [form.cleaned_data['cars']],
                'Landsize': [form.cleaned_data['landsize']],
                'CouncilArea': [form.cleaned_data['council_area']],
                'Lattitude': [form.cleaned_data['latitude']],
                'Longtitude': [form.cleaned_data['longitude']],
                'Regionname': [form.cleaned_data['region']],
                'Propertycount': [form.cleaned_data['property_count']]
            }
            
            # Create a DataFrame with the input data
            input_df = pd.DataFrame(input_data)
            
            try:
                # Make prediction
                prediction = model.predict(input_df)
                predicted_price = prediction[0]
                
                return render(request, 'predictor/result.html', {
                    'predicted_price': f"${predicted_price:,.2f}",
                    'form_data': form.cleaned_data
                })
            except Exception as e:
                return render(request, 'predictor/error.html', {
                    'error_message': f'Error making prediction: {str(e)}'
                })
    else:
        form = PredictionForm()
    
    return render(request, 'predictor/home.html', {'form': form})

def about(request):
    """About page with information about the project"""
    return render(request, 'predictor/about.html')
