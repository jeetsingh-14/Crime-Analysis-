import pandas as pd
import numpy as np
import os

# Step 1: Load the cleaned dataset
# Adjust the path based on where your cleaned file is located
# Example: if your cleaned file is saved under "data/cleaned/full_cleaned.csv"

file_path = r"Data/processed/cleaned_full_data.csv"  # <-- CHANGE this to your actual path

crime_data = pd.read_csv(file_path)

print("✅ Loaded cleaned dataset with shape:", crime_data.shape)

# Step 2: Feature Engineering

# Ensure Date is datetime type
crime_data['Date_Occurred'] = pd.to_datetime(crime_data['Date_Occurred'], errors='coerce')

# Extract Year, Month, Day of Week
crime_data['Year_Occurred'] = crime_data['Date_Occurred'].dt.year
crime_data['Month_Occurred'] = crime_data['Date_Occurred'].dt.month
crime_data['DayOfWeek_Occurred'] = crime_data['Date_Occurred'].dt.day_name()

# Extract Hour from Time_Occurred
crime_data['Hour_Occurred'] = (crime_data['Time_Occurred'] // 100).astype('Int64')

# Create Night/Day Feature
def is_night(hour):
    if pd.isnull(hour):
        return 0
    return 1 if (hour >= 20 or hour <= 6) else 0

crime_data['Is_Night'] = crime_data['Hour_Occurred'].apply(is_night)

# Simplify Premise
def simplify_premise(premise):
    if pd.isnull(premise):
        return 'Other'
    premise = premise.upper()
    if 'SINGLE FAMILY' in premise or 'APARTMENT' in premise or 'HOUSE' in premise:
        return 'Residence'
    elif 'STREET' in premise or 'PARKING' in premise:
        return 'Street/Parking'
    elif 'STORE' in premise or 'SHOP' in premise or 'MARKET' in premise:
        return 'Store'
    elif 'SCHOOL' in premise or 'COLLEGE' in premise:
        return 'School'
    else:
        return 'Other'

crime_data['Premise_Type'] = crime_data['Premise_Description'].apply(simplify_premise)

# Create Weapon Used Flag
crime_data['Weapon_Used_Flag'] = crime_data['Weapon_Description'].apply(lambda x: 0 if pd.isnull(x) else 1)

# Victim Age Group
def age_group(age):
    if pd.isnull(age):
        return 'Unknown'
    elif age <= 12:
        return 'Child'
    elif age <= 19:
        return 'Teen'
    elif age <= 59:
        return 'Adult'
    else:
        return 'Senior'

crime_data['Victim_Age_Group'] = crime_data['Victim_Age'].apply(age_group)

print("✅ Feature Engineering completed.")
print("Current Columns:", crime_data.columns.tolist())

# Optional: Save again if you want
output_path = r"Data/processed/feature_engineered_data.csv"
crime_data.to_csv(output_path, index=False)
