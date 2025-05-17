# ml_modeling.py
"""
Los Angeles Crime Prediction Model
This script builds machine learning models to predict crime categories and types
based on various features including location, time, and victim demographics.
It also identifies popular crime locations and predicts crime types for these locations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Ensure output directory exists
os.makedirs("outputs_1", exist_ok=True)


# Step 1: Load the feature-engineered dataset
file_path = r"Data/processed/feature_engineered_data.csv"  # <-- update your real path here
crime_data = pd.read_csv(file_path)

print("✅ Loaded dataset with shape:", crime_data.shape)

# Step 2: Define X (features) and y (target)
# Drop unnecessary or problematic columns including 'Modus_Operandi'
# Step 2: Define X (features) and y (target)
X = crime_data.drop([
    'ID', 'Crime_Category', 'Crime_Description',
    'Date_Occurred', 'Date_Reported',
    'Location', 'Cross_Street',
    'Modus_Operandi',
    'Premise_Description',
    'Weapon_Description',
    'Status',                # <-- Drop this new
    'Status_Description'      # <-- Drop this new
], axis=1)


y = crime_data['Crime_Category']

# Step 3: Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print(f"✅ Split complete: Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Step 4: Setup preprocessing
categorical_features = [
    'DayOfWeek_Occurred',
    'Premise_Type',
    'Victim_Age_Group',
    'Area_Name',
    'Victim_Sex',
    'Victim_Descent'    # <-- ADD THIS
]
numeric_features = ['Latitude', 'Longitude', 'Victim_Age', 'Hour_Occurred', 'Is_Night', 'Weapon_Used_Flag']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numeric with mean
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing categorical with most frequent value
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Combine into column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
    # NO remainder='passthrough' anymore
)

# Step 5: Define base model
model = RandomForestClassifier(random_state=42)

# Step 6: Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Step 7: Train the base model
pipeline.fit(X_train, y_train)

print("✅ Base model training complete.")

# Step 8: Evaluate base model
y_pred = pipeline.predict(X_test)

print("\nClassification Report (Base Model):")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix (Base Model):")
print(confusion_matrix(y_test, y_pred))

# Step 9: Setup GridSearchCV for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='f1_macro'  # Important for multiclass
)

# Step 10: Run GridSearch
print("\n🔍 Starting Grid Search...")
grid_search.fit(X_train, y_train)

# Step 11: Best model results
print("✅ Grid Search complete.")
print("✅ Best Parameters:", grid_search.best_params_)
print("✅ Best Cross-Validated Score:", grid_search.best_score_)

# Step 12: Evaluate best model
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print("\nClassification Report (Tuned Best Model):")
print(classification_report(y_test, y_pred_best))

print("\nConfusion Matrix (Tuned Best Model):")
print(confusion_matrix(y_test, y_pred_best))

# ===============================================================
# Part 2: Popular Locations and Crime Type Prediction
# ===============================================================

print("\n\n🔍 Analyzing Popular Crime Locations and Predicting Crime Types...")

# Step 1: Identify popular locations based on crime frequency
location_crime_counts = crime_data.groupby('Area_Name').size().reset_index(name='Crime_Count')
location_crime_counts = location_crime_counts.sort_values('Crime_Count', ascending=False)

# Get top 10 locations with highest crime counts
top_locations = location_crime_counts.head(10)
print("\n✅ Top 10 Locations by Crime Frequency:")
print(top_locations)

# Save the top locations to a CSV file
top_locations.to_csv("outputs_1/top_crime_locations.csv", index=False)

# Step 2: Analyze crime types in popular locations
print("\n🔍 Analyzing Crime Types in Popular Locations...")

# Create a dataframe to store crime type predictions for each location
location_predictions = pd.DataFrame()

# For each top location, analyze and predict crime types
for location in top_locations['Area_Name']:
    print(f"\n🔍 Analyzing {location}...")

    # Filter data for this location
    location_data = crime_data[crime_data['Area_Name'] == location]

    # Count crime types in this location
    crime_type_counts = location_data['Crime_Category'].value_counts().reset_index()
    crime_type_counts.columns = ['Crime_Type', 'Count']
    crime_type_counts['Percentage'] = (crime_type_counts['Count'] / crime_type_counts['Count'].sum()) * 100

    # Get top 5 crime types for this location
    top_crime_types = crime_type_counts.head(5)
    print(f"Top Crime Types in {location}:")
    print(top_crime_types)

    # Add to location predictions dataframe
    location_row = {'Location': location, 'Total_Crimes': len(location_data)}
    for _, row in top_crime_types.iterrows():
        location_row[f"Crime_Type_{_+1}"] = row['Crime_Type']
        location_row[f"Crime_Count_{_+1}"] = row['Count']
        location_row[f"Crime_Percent_{_+1}"] = row['Percentage']

    location_predictions = pd.concat([location_predictions, pd.DataFrame([location_row])], ignore_index=True)

# Save location crime type analysis
location_predictions.to_csv("outputs_1/location_crime_type_analysis.csv", index=False)
print("\n✅ Location Crime Type Analysis saved to outputs_1/location_crime_type_analysis.csv")

# Step 3: Build predictive models for crime types by time for each popular location
print("\n🔍 Building Predictive Models for Crime Types by Time...")

# Create a function to build and evaluate a model for a specific location
def build_location_crime_model(location_name):
    """Build a model to predict crime counts by hour for a specific location."""
    # Filter data for this location
    location_data = crime_data[crime_data['Area_Name'] == location_name]

    if len(location_data) < 100:  # Skip if not enough data
        print(f"⚠️ Not enough data for {location_name} to build a reliable model")
        return None

    # Group by hour and crime type to get counts
    hour_crime_counts = location_data.groupby(['Hour_Occurred', 'Crime_Category']).size().reset_index(name='Count')

    # Pivot to get each crime type as a column
    hour_crime_pivot = hour_crime_counts.pivot(index='Hour_Occurred', columns='Crime_Category', values='Count').reset_index()
    hour_crime_pivot = hour_crime_pivot.fillna(0)

    # Prepare features (X) and target (y) - predict all crime types based on hour
    X = hour_crime_pivot[['Hour_Occurred']]

    # Get the most common crime types
    top_crime_types = location_data['Crime_Category'].value_counts().head(3).index.tolist()

    # Create models for each top crime type
    models = {}
    predictions = {}

    for crime_type in top_crime_types:
        if crime_type in hour_crime_pivot.columns:
            y = hour_crime_pivot[crime_type]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"  ✅ Model for {crime_type} in {location_name}: MSE={mse:.2f}, R²={r2:.2f}")

            # Store model
            models[crime_type] = model

            # Make predictions for all hours
            all_hours = pd.DataFrame({'Hour_Occurred': range(24)})
            predictions[crime_type] = model.predict(all_hours)

    return {
        'location': location_name,
        'models': models,
        'predictions': predictions,
        'top_crime_types': top_crime_types
    }

# Build models for each top location
location_models = {}
for location in top_locations['Area_Name']:
    model_result = build_location_crime_model(location)
    if model_result:
        location_models[location] = model_result

# Step 4: Visualize predictions for each location
print("\n🔍 Creating Visualizations for Crime Predictions...")

for location, model_data in location_models.items():
    # Create a figure for this location
    plt.figure(figsize=(12, 8))

    # Plot predictions for each crime type
    for crime_type in model_data['top_crime_types']:
        if crime_type in model_data['predictions']:
            plt.plot(range(24), model_data['predictions'][crime_type], label=crime_type)

    plt.title(f'Predicted Hourly Crime Counts for {location}')
    plt.xlabel('Hour of Day')
    plt.ylabel('Predicted Crime Count')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Save the figure
    output_path = f"outputs_1/crime_prediction_{location.replace(' ', '_')}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Prediction visualization for {location} saved to {output_path}")

# Step 5: Create a summary report
print("\n🔍 Creating Summary Report...")

# Create a summary dataframe
summary = pd.DataFrame()

for location, model_data in location_models.items():
    # For each location, find the hour with highest predicted crime for each type
    for crime_type in model_data['top_crime_types']:
        if crime_type in model_data['predictions']:
            predictions = model_data['predictions'][crime_type]
            peak_hour = np.argmax(predictions)
            peak_count = predictions[peak_hour]

            # Add to summary
            summary = pd.concat([summary, pd.DataFrame([{
                'Location': location,
                'Crime_Type': crime_type,
                'Peak_Hour': peak_hour,
                'Peak_Predicted_Count': peak_count,
                'Risk_Level': 'High' if peak_count > 10 else ('Medium' if peak_count > 5 else 'Low')
            }])], ignore_index=True)

# Save summary report
summary.to_csv("outputs_1/crime_prediction_summary.csv", index=False)
print("\n✅ Crime Prediction Summary saved to outputs_1/crime_prediction_summary.csv")

print("\n✅ All crime prediction analysis completed and saved to outputs_1/ directory")
