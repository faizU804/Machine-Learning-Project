# Melbourne Housing Price Prediction

This project provides a comprehensive machine learning solution for predicting house prices in Melbourne using various regression models.

## Project Overview

This machine learning project:

1. Loads and explores the Melbourne housing dataset
2. Cleans and preprocesses the data
3. Trains multiple regression models (Linear Regression, Random Forest, Gradient Boosting)
4. Evaluates model performance using RMSE and R²
5. Visualizes results and saves the best model

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository or download the files
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the `melb_data.csv` file is in the root directory
2. Run the main script:

```bash
python melbourne_housing_prediction.py
```

3. Check the `output` directory for:
   - Visualization plots
   - The saved best model (best_model.pkl)
   - Performance metrics

## Project Structure

- `melbourne_housing_prediction.py`: Main script with the complete ML pipeline
- `melb_data.csv`: Melbourne housing dataset
- `requirements.txt`: Required Python packages
- `output/`: Directory containing results and visualizations
  - `price_distribution.png`: Distribution of house prices
  - `correlation_matrix.png`: Correlation between features
  - `price_by_rooms.png`: Price distribution by number of rooms
  - `rmse_comparison.png`: RMSE comparison across models
  - `r2_comparison.png`: R² comparison across models
  - `actual_vs_predicted.png`: Actual vs Predicted prices for best model
  - `feature_importance.png`: Feature importance for the best model (if applicable)
  - `best_model.pkl`: Serialized best model

## Models Used

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Features

The project handles:
- Missing value imputation
- Categorical feature encoding
- Feature importance analysis
- Model comparison and selection 