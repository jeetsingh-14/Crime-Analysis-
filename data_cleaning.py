import pandas as pd

# Step 1: Load the original LAPD dataset
crime_data = pd.read_csv("Data/unprocessed/Crime_Data_from_2020_to_Present.csv")

# Select and rename important columns
columns_to_keep = [
    'LOCATION', 'Cross Street', 'LAT', 'LON',
    'Date Rptd', 'DATE OCC', 'TIME OCC',
    'AREA', 'AREA NAME', 'Rpt Dist No',
    'Part 1-2', 'Mocodes', 'Vict Age', 'Vict Sex', 'Vict Descent',
    'Premis Cd', 'Premis Desc',
    'Weapon Used Cd', 'Weapon Desc',
    'Status', 'Status Desc', 'Crm Cd Desc'  # <-- Also keeping Crime Description!
]

crime_data = crime_data[columns_to_keep]

# Rename columns
column_rename_mapping = {
    'LOCATION': 'Location',
    'Cross Street': 'Cross_Street',
    'LAT': 'Latitude',
    'LON': 'Longitude',
    'Date Rptd': 'Date_Reported',
    'DATE OCC': 'Date_Occurred',
    'TIME OCC': 'Time_Occurred',
    'AREA': 'Area_ID',
    'AREA NAME': 'Area_Name',
    'Rpt Dist No': 'Reporting_District_no',
    'Part 1-2': 'Part 1-2',
    'Mocodes': 'Modus_Operandi',
    'Vict Age': 'Victim_Age',
    'Vict Sex': 'Victim_Sex',
    'Vict Descent': 'Victim_Descent',
    'Premis Cd': 'Premise_Code',
    'Premis Desc': 'Premise_Description',
    'Weapon Used Cd': 'Weapon_Used_Code',
    'Weapon Desc': 'Weapon_Description',
    'Status': 'Status',
    'Status Desc': 'Status_Description',
    'Crm Cd Desc': 'Crime_Description'
}
crime_data = crime_data.rename(columns=column_rename_mapping)

# Convert numeric fields safely
crime_data['Victim_Age'] = pd.to_numeric(crime_data['Victim_Age'], errors='coerce')
crime_data['Time_Occurred'] = pd.to_numeric(crime_data['Time_Occurred'], errors='coerce')

# Stronger crime category mapping based on Crime Description
def enhanced_map_crime_category(desc):
    if pd.isnull(desc):
        return 'Other Crimes'
    desc = desc.upper()
    if any(x in desc for x in ['ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE', 'MURDER', 'RAPE', 'SEXUAL']):
        return 'Crimes against Persons'
    elif any(x in desc for x in ['BURGLARY', 'THEFT', 'GRAND THEFT', 'PETTY THEFT', 'ARSON', 'VANDALISM', 'MOTOR VEHICLE']):
        return 'Property Crimes'
    elif any(x in desc for x in ['WEAPONS', 'NARCOTICS', 'DISORDERLY', 'ALCOHOL', 'GAMBLING']):
        return 'Crimes against Public Order'
    else:
        return 'Other Crimes'

# Apply the new crime mapping
crime_data['Crime_Category'] = crime_data['Crime_Description'].apply(enhanced_map_crime_category)

# Create ID
crime_data = crime_data.reset_index(drop=True)
crime_data['ID'] = crime_data.index + 1

# Create sample submission
sample_submission = crime_data[['ID', 'Crime_Category']].copy()

# Final
print("✅ Full cleaned dataset 'crime_data' ready with enhanced crime mapping.")
print("✅ Sample submission 'sample_submission' ready.")

# Optional: if you want to save them
crime_data.to_csv("./Data/processed/cleaned_full_data.csv", index=False)
sample_submission.to_csv("./Data/processed/sample_file.csv", index=False)