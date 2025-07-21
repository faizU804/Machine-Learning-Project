#!/usr/bin/env python
# Melbourne Housing Price Prediction

# Step 1: Setup Environment - Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import warnings
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Step 2: Load the Dataset
print("Loading dataset...")
data = pd.read_csv('melb_data.csv')

# Step 3: Explore the Data (EDA)
print("\n--- Dataset Overview ---")
print(f"Shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

print("\nData Types:")
print(data.dtypes)

print("\nMissing values count:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

# Create output directory for visualizations
os.makedirs('output', exist_ok=True)

# Visualize the distribution of the target variable (Price)
plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.savefig('output/price_distribution.png')
plt.close()

# Correlation Matrix for numerical features
plt.figure(figsize=(12, 10))
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('output/correlation_matrix.png')
plt.close()

# Boxplot for Price by Rooms
plt.figure(figsize=(12, 6))
sns.boxplot(x='Rooms', y='Price', data=data)
plt.title('Price Distribution by Number of Rooms')
plt.savefig('output/price_by_rooms.png')
plt.close()

# Step 4: Data Cleaning
print("\n--- Data Cleaning ---")

# Check the percentage of missing values in each column
missing_percentage = data.isnull().mean() * 100
print("\nPercentage of missing values:")
print(missing_percentage[missing_percentage > 0])

# Remove columns with high percentage of missing values (e.g., > 30%)
high_missing_cols = missing_percentage[missing_percentage > 30].index.tolist()
if high_missing_cols:
    print(f"\nRemoving columns with high missing values: {high_missing_cols}")
    data = data.drop(columns=high_missing_cols)

# Drop columns that are not useful for prediction (based on feature importance and domain knowledge)
columns_to_keep = [
    'Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Landsize',
    'Lattitude', 'Longtitude', 'Propertycount', 'Type', 'Regionname'
]
columns_to_drop = [col for col in data.columns if col not in columns_to_keep + ['Price']]
if columns_to_drop:
    print(f"\nDropping unnecessary columns: {columns_to_drop}")
    data = data.drop(columns=columns_to_drop)

# Step 5: Feature Engineering and Selection
print("\n--- Feature Engineering and Selection ---")

# Identify categorical and numerical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('Price')  # Remove target variable from features

print(f"\nCategorical columns: {categorical_cols}")
print(f"\nNumerical columns: {numerical_cols}")

# Step 6: Define Features and Target
print("\n--- Defining Features and Target ---")
y = data['Price']  # Target
X = data.drop('Price', axis=1)

# Step 7: Split the Dataset
print("\n--- Splitting the Dataset ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Step 8: Create Preprocessing Pipeline
print("\n--- Creating Preprocessing Pipeline ---")

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 9: Train and Evaluate Multiple Models
print("\n--- Training and Evaluating Models ---")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'KNN': Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', SimpleImputer(strategy='median'))
            ]), numerical_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'KNN':
        # KNN pipeline already includes preprocessing
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        model = pipeline
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - RMSE: ${rmse:.2f}, R²: {r2:.4f}")
    results[name] = {
        'pipeline': model,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }

# Step 10: Visualize Model Performance
print("\n--- Visualizing Model Performance ---")

# Compare RMSE of all models
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [results[model]['rmse'] for model in results])
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/rmse_comparison.png')
plt.close()

# Compare R² of all models
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [results[model]['r2'] for model in results])
plt.title('R² Comparison')
plt.ylabel('R²')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/r2_comparison.png')
plt.close()

# Step 11: Save the Best Model
print("\n--- Saving the Best Model ---")

# Find the best model based on R² score
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['pipeline']
best_rmse = results[best_model_name]['rmse']
best_r2 = results[best_model_name]['r2']

print(f"Best Model: {best_model_name}")
print(f"RMSE: ${best_rmse:.2f}")
print(f"R²: {best_r2:.4f}")

# --- AUTOMATIC FEATURE SELECTION BASED ON IMPORTANCE ---
# Only for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n--- Automatic Feature Selection ---")
    cat_features = best_model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_features)
    importances = best_model.named_steps['model'].feature_importances_
    # Select features with importance >= 0.01
    selected_indices = [i for i, imp in enumerate(importances) if imp >= 0.01]
    selected_features = [feature_names[i] for i in selected_indices]
    print(f"Selected features (importance >= 0.01): {selected_features}")
    # Save selected features for web interface
    with open('output/selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    # Reduce X to only important features (original columns)
    # For categorical, keep only those whose one-hot columns are selected
    keep_cats = set()
    for feat in selected_features:
        for cat in categorical_cols:
            if feat.startswith(cat + '_'):
                keep_cats.add(cat)
    reduced_categorical_cols = [cat for cat in categorical_cols if cat in keep_cats]
    reduced_numerical_cols = [col for col in numerical_cols if col in selected_features]
    print(f"Reduced categorical columns: {reduced_categorical_cols}")
    print(f"Reduced numerical columns: {reduced_numerical_cols}")
    # Rebuild X with only these columns
    X_reduced = X[reduced_numerical_cols + reduced_categorical_cols]
    # Retrain models on reduced features
    print("\n--- Retraining Models on Selected Features ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    # Tune hyperparameters for higher accuracy
    tuned_models = {
        'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=4, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
        'Linear Regression': LinearRegression(),
        'KNN': Pipeline([
            ('preprocessor', ColumnTransformer([
                ('num', Pipeline([
                    ('scaler', StandardScaler()),
                    ('imputer', SimpleImputer(strategy='median'))
                ]), reduced_numerical_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), reduced_categorical_cols)
            ])),
            ('model', KNeighborsRegressor(n_neighbors=5))
        ])
    }
    tuned_results = {}
    for name, model in tuned_models.items():
        print(f"\nTraining {name} (tuned)...")
        if name == 'KNN':
            model.fit(X_train_r, y_train_r)
            y_pred = model.predict(X_test_r)
        else:
            pipeline = Pipeline(steps=[
                ('preprocessor', ColumnTransformer([
                    ('num', SimpleImputer(strategy='median'), reduced_numerical_cols),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), reduced_categorical_cols)
                ])),
                ('model', model)
            ])
            pipeline.fit(X_train_r, y_train_r)
            y_pred = pipeline.predict(X_test_r)
            model = pipeline
        mse = mean_squared_error(y_test_r, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_r, y_pred)
        print(f"{name} (tuned) - RMSE: ${rmse:.2f}, R²: {r2:.4f}")
        tuned_results[name] = {
            'pipeline': model,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
    # Save the best tuned model
    best_tuned_name = max(tuned_results, key=lambda x: tuned_results[x]['r2'])
    best_tuned_model = tuned_results[best_tuned_name]['pipeline']
    best_tuned_rmse = tuned_results[best_tuned_name]['rmse']
    best_tuned_r2 = tuned_results[best_tuned_name]['r2']
    print(f"\nBest Tuned Model: {best_tuned_name}")
    print(f"Final RMSE: ${best_tuned_rmse:.2f}")
    print(f"Final R² (accuracy): {best_tuned_r2:.4f}")
    # Save the tuned model
    with open('output/best_model.pkl', 'wb') as f:
        pickle.dump(best_tuned_model, f)
    # Save the final accuracy
    with open('output/model_accuracy.txt', 'w') as f:
        f.write(f"R2: {best_tuned_r2:.4f}\nRMSE: ${best_tuned_rmse:.2f}\n")
    print("Best tuned model and accuracy saved.")
else:
    # Save the model and accuracy as before
    with open('output/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('output/model_accuracy.txt', 'w') as f:
        f.write(f"R2: {best_r2:.4f}\nRMSE: ${best_rmse:.2f}\n")
    print("Best model and accuracy saved.")

# Step 12: Actual vs Predicted Plot for the Best Model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results[best_model_name]['predictions'], alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs Predicted Prices ({best_model_name})')
plt.tight_layout()
plt.savefig('output/actual_vs_predicted.png')
plt.close()

# Step 13: Feature Importance (if applicable)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n--- Feature Importance Analysis ---")
    
    # Get feature names after preprocessing
    cat_features = best_model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_features)
    
    # Get feature importances
    importances = best_model.named_steps['model'].feature_importances_
    
    # Sort them
    indices = np.argsort(importances)[-15:]  # Top 15 features
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Features')
    plt.tight_layout()
    plt.savefig('output/feature_importance.png')
    plt.close()

print("\n--- Analysis Complete ---")
print("All results and visualizations have been saved to the 'output' folder.") 